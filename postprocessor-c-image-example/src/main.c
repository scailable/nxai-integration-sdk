// Standard system includes
#include <signal.h>
#include <stdio.h>
#include <string.h>

// Local includes
#include "mpack.h"
#include "nxai_socket_utils.h"

// Deps includes
#include "data_utils.h"
#include "nxai_data_structures.h"
#include "nxai_shm_utils.h"

#define STRLEN( s ) ( sizeof( s ) / sizeof( s[0] ) )

// Flag to keep track of interrupts
volatile sig_atomic_t interrupt_flag = false;

uint64_t processInputTensor( const char *input_buffer, size_t message_length );

char *processMpackDocument( const char *input_buffer, size_t input_buffer_length, size_t *output_buffer_length, char *image_header, size_t header_length );
/**
 * @brief Function to handle interrupt signals
 *
 * When the NXAI Runtime is stopped, it sends an interrupt signal to all postprocessors it started
 * and waits for them to exit.
 *
 * This function sets a global interrupt flag which stops the socket listener loop. After the loop is broken
 * the socket is closed and the application exits gracefully.
 *
 * @param sig
 */
void handle_interrupt( int sig );

// Main function, the entry point of the application
int main( int argc, char *argv[] ) {
    // Check for the correct number of command line arguments
    if ( argc != 2 ) {
        printf( "EXAMPLE POSTPROCESSOR: Started with incorrect command line arguments.\n" );
        exit( 2 );
    }

    // Initialize socket system
    nxai_socket_initialize_sockets();

    // Set signal handler to listen for interrupt signals
    signal( SIGINT, handle_interrupt );

    const char *socket_path = argv[1];

    printf( "EXAMPLE POSTPROCESSOR: Starting up with socket path: %s.\n", socket_path );

    char *input_buffer = NULL;
    size_t allocated_buffer_size = 0;
    uint32_t message_length;

    // Create a listener socket
    nxai_socket_t socket_fd = nxai_socket_create_listener( socket_path );

    // Main loop: continues until an interrupt signal is received
    while ( interrupt_flag == false ) {
        // Wait for a message on the socket
        nxai_socket_t connection_fd = nxai_socket_await_message( socket_fd, &allocated_buffer_size, &input_buffer, &message_length );

        // If connection times out, it continues to wait for the message again
        if ( connection_fd == -1 ) {
            continue;
        }

        // Since we're expecting input tensor, read data header
        char *image_header = NULL;
        uint32_t header_length = 0;
        nxai_socket_receive_on_connection( connection_fd, &allocated_buffer_size, &image_header, &header_length );
        printf( "EXAMPLE PLUGIN: Received header %s\n", input_buffer );

        // Process the Mpack document
        size_t output_length;
        char *output_message = processMpackDocument( input_buffer, message_length, &output_length, image_header, header_length );

        // Send the processed output back to the socket
        nxai_socket_send_to_connection( connection_fd, output_message, ( uint32_t ) output_length );

        // Free buffer
        free( output_message );

        // Close the connection
        if ( nxai_close_socket( connection_fd ) == -1 ) {
            fprintf( stderr, "EXAMPLE POSTPROCESSOR: Warning: Sender socket close error!\n" );
        }
    }

    // Unlink socket file so it can be used again
    unlink( socket_path );

    printf( "EXAMPLE POSTPROCESSOR: Exiting.\n" );
}

/**
 * @brief Processes an MPack document, manipulates the data, and writes the output.
 *
 * This function takes an input buffer containing an MPack document, parses it to extract various pieces of data (such as timestamp, image dimensions, counts, scores, and bounding boxes), manipulates the data (e.g., adding a test bounding box), and then writes the manipulated data back into an output buffer.
 *
 * @param input_buffer A pointer to the input buffer containing the MPack document.
 * @param input_buffer_length The length of the input buffer.
 * @param output_buffer_length A pointer to a size_t variable where the length of the output buffer will be stored.
 * @return A pointer to the output buffer containing the manipulated MPack document.
 */
char *processMpackDocument( const char *input_buffer, size_t input_buffer_length, size_t *output_buffer_length, char *image_header, size_t header_length ) {

    ////////////////////////////////////////////////////
    //// Parse input data
    ////////////////////////////////////////////////////

    // Initialize Node API tree
    mpack_tree_t tree;
    mpack_tree_init_data( &tree, input_buffer, input_buffer_length );
    mpack_tree_parse( &tree );
    mpack_node_t inference_results_root = mpack_tree_root( &tree );

    // Read timestamp
    mpack_node_t timestamp_node = mpack_node_map_cstr( inference_results_root, "Timestamp" );
    uint64_t timestamp = mpack_node_u64( timestamp_node );

    uint32_t image_width = mpack_node_u32( mpack_node_map_cstr( inference_results_root, "Width" ) );
    uint32_t image_height = mpack_node_u32( mpack_node_map_cstr( inference_results_root, "Height" ) );
    uint32_t input_index = mpack_node_u32( mpack_node_map_cstr( inference_results_root, "InputIndex" ) );

    // Read counts
    count_object_t *counts = NULL;
    size_t num_counts = 0;
    mpack_node_t counts_node = mpack_node_map_cstr_optional( inference_results_root, "Counts" );
    if ( mpack_node_is_missing( counts_node ) == false ) {
        // Parse counts
        num_counts = mpack_node_map_count( counts_node );
        if ( num_counts != 0 ) {
            counts = malloc( sizeof( count_object_t ) * num_counts );
        }
        for ( size_t counts_index = 0; counts_index < num_counts; counts_index++ ) {
            uint32_t count = mpack_node_u32( mpack_node_map_value_at( counts_node, counts_index ) );
            char *count_class = mpack_node_cstr_alloc( mpack_node_map_key_at( counts_node, counts_index ), 1024 );
            counts[counts_index] = (count_object_t) { .class_name = count_class, .count = count };
        }
    }

    // Read scores
    score_object_t *scores = NULL;
    size_t num_scores = 0;
    mpack_node_t scores_node = mpack_node_map_cstr_optional( inference_results_root, "Scores" );
    if ( mpack_node_is_missing( scores_node ) == false ) {
        // Parse scores
        num_scores = mpack_node_map_count( scores_node );
        if ( num_scores != 0 ) {
            scores = malloc( sizeof( score_object_t ) * num_scores );
        }
        for ( size_t scores_index = 0; scores_index < num_scores; scores_index++ ) {
            float score = mpack_node_float( mpack_node_map_value_at( scores_node, scores_index ) );
            char *score_class = mpack_node_cstr_alloc( mpack_node_map_key_at( scores_node, scores_index ), 1024 );
            scores[scores_index] = (score_object_t) { .class_name = score_class, .score = score };
        }
    }

    // Read bboxes
    bbox_object_t *bboxs = NULL;
    size_t num_bboxs = 0;
    mpack_node_t bboxs_node = mpack_node_map_cstr_optional( inference_results_root, "BBoxes_xyxy" );
    if ( mpack_node_is_missing( bboxs_node ) == false ) {
        // Parse bboxs
        num_bboxs = mpack_node_map_count( bboxs_node );
        if ( num_bboxs != 0 ) {
            bboxs = malloc( sizeof( bbox_object_t ) * num_bboxs );
        }
        for ( size_t bboxs_index = 0; bboxs_index < num_bboxs; bboxs_index++ ) {
            mpack_node_t coordinates_data_node = mpack_node_map_value_at( bboxs_node, bboxs_index );
            const char *bin_data = mpack_node_bin_data( coordinates_data_node );
            size_t bin_size = mpack_node_bin_size( coordinates_data_node );
            // Copy data to ensure alignment
            float *coordinates = (float *) malloc( bin_size );
            memcpy( coordinates, bin_data, bin_size );
            char *bbox_class = mpack_node_cstr_alloc( mpack_node_map_key_at( bboxs_node, bboxs_index ), 1024 );
            bboxs[bboxs_index] = (bbox_object_t) { .class_name = bbox_class, .coordinates = coordinates, .format = "xyxy", .coords_length = bin_size / sizeof( float ) };
        }
    }

    ////////////////////////////////////////////////////
    //// Manipulate data
    ////////////////////////////////////////////////////
    // Suppress unused variable warnings
    (void) input_index;
    (void) image_height;
    (void) image_width;
    (void) timestamp;

    // Count pixel values of input tensor
    uint64_t cumulative_count = processInputTensor( image_header, header_length );
    num_counts++;
    counts = realloc( counts, sizeof( count_object_t ) * num_counts );
    counts[num_counts - 1] = (count_object_t) { .class_name = strdup( "ImageBytesCumalitive" ), .count = cumulative_count };

    ////////////////////////////////////////////////////
    //// Write output data
    ////////////////////////////////////////////////////

    size_t buffer_size;
    char *mpack_buffer = nxai_write_buffer( inference_results_root, num_bboxs, bboxs, num_scores, scores, num_counts, counts, &buffer_size );

    *output_buffer_length = buffer_size;

    ////////////////////////////////////////////////////
    //// Clean up data
    ////////////////////////////////////////////////////

    for ( size_t index = 0; index < num_bboxs; index++ ) {
        free( bboxs[index].class_name );
        free( bboxs[index].coordinates );
        free( bboxs[index].confidences );
    }
    free( bboxs );

    // Free counts
    for ( size_t index = 0; index < num_counts; index++ ) {
        free( counts[index].class_name );
    }
    free( counts );

    // Free scores
    for ( size_t index = 0; index < num_scores; index++ ) {
        free( scores[index].class_name );
    }
    free( scores );

    return mpack_buffer;
}

uint64_t processInputTensor( const char *input_buffer, size_t message_length ) {

    // Initialize Node API tree
    mpack_tree_t tree;
    mpack_tree_init_data( &tree, input_buffer, message_length );
    mpack_tree_parse( &tree );
    mpack_node_t tensor_header_root = mpack_tree_root( &tree );

    // Get SHM key
    char *shm_key_string = mpack_node_cstr_alloc( mpack_node_map_cstr( tensor_header_root, "SHMKEY" ), 1024 );
    // Get SHM details
    nxai_shm_t input_shm;
    nxai_shm_key_from_string( &input_shm, shm_key_string );
    nxai_shm_get_id( &input_shm );

    size_t tensor_size;
    char *tensor_data;
    void *shared_data = nxai_shm_read( &input_shm, &tensor_size, &tensor_data );

    uint32_t height = mpack_node_u32( mpack_node_map_cstr( tensor_header_root, "Height" ) );
    uint32_t width = mpack_node_u32( mpack_node_map_cstr( tensor_header_root, "Width" ) );
    uint32_t channels = mpack_node_u32( mpack_node_map_cstr( tensor_header_root, "Channels" ) );

    size_t actual_size = height * width * channels;
    printf( "EXAMPLE PLUGIN: Read size: %zu Actual size: %zu\n", tensor_size, actual_size );

    // Add all pixel values
    uint64_t cumulative_count = 0;
    for ( size_t index = 0; index < tensor_size; index++ ) {
        cumulative_count += (unsigned char) tensor_data[index];
    }

    nxai_shm_close( shared_data );

    return cumulative_count;
}

// Function to handle interrupt signals
void handle_interrupt( int signal ) {
    (void) signal;
    // Set the interrupt flag to true on receiving an interrupt signal
    interrupt_flag = true;
}
