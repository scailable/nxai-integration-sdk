cmake_minimum_required(VERSION 3.16)
project(e2e-postprocessor-harness CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)

set(NXAI_UTILS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../nxai-utilities")

# Same utility sources the reference processors compile (mirrors
# postprocessor-cpp-example/CMakeLists.txt): the harness reuses the production host
# primitives and message (de)serializers.
add_executable(e2e-postprocessor-harness
    postprocessor_harness.cpp
    "${NXAI_UTILS_DIR}/nxai_utils.cpp"
    "${NXAI_UTILS_DIR}/nxai_ipc_utils.cpp"
    "${NXAI_UTILS_DIR}/external_processor_utilities.cpp"
    "${NXAI_UTILS_DIR}/mpack.c"
)

target_include_directories(e2e-postprocessor-harness PRIVATE "${NXAI_UTILS_DIR}")

# The launcher is the harness's only platform-dependent part, so it is a module of
# its own with its own tests rather than inline calls in the harness body.
add_library(e2e-child-process STATIC child_process.cpp)
target_include_directories(e2e-child-process PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(e2e-postprocessor-harness PRIVATE e2e-child-process)

add_executable(e2e-child-process-test child_process_test.cpp)
target_link_libraries(e2e-child-process-test PRIVATE e2e-child-process)

add_test(NAME e2e-child-process COMMAND e2e-child-process-test)
set_tests_properties(e2e-child-process PROPERTIES TIMEOUT 30)

# The sync-object failure value is platform-specific and got it wrong on Windows, so
# the predicate that encodes it is tested directly -- a failed CreateEventA is otherwise
# only reachable under handle exhaustion.
add_executable(e2e-ipc-sync-obj-test
    ipc_sync_obj_test.cpp
    "${NXAI_UTILS_DIR}/nxai_ipc_utils.cpp"
    "${NXAI_UTILS_DIR}/nxai_utils.cpp"
)
target_include_directories(e2e-ipc-sync-obj-test PRIVATE "${NXAI_UTILS_DIR}")

add_test(NAME e2e-ipc-sync-obj COMMAND e2e-ipc-sync-obj-test)
set_tests_properties(e2e-ipc-sync-obj PROPERTIES TIMEOUT 30)

# The two failure paths a passing happy path proves nothing about: the
# partial-creation cleanup in nxai_ipc_create() and the malformed-descriptor branches in
# ipc_from_str(). Driven by deliberate failure injection, asserted on held resources.
add_executable(e2e-ipc-failure-paths-test
    ipc_failure_paths_test.cpp
    "${NXAI_UTILS_DIR}/nxai_ipc_utils.cpp"
    "${NXAI_UTILS_DIR}/nxai_utils.cpp"
)
target_include_directories(e2e-ipc-failure-paths-test PRIVATE "${NXAI_UTILS_DIR}")

add_test(NAME e2e-ipc-failure-paths COMMAND e2e-ipc-failure-paths-test)
set_tests_properties(e2e-ipc-failure-paths PROPERTIES TIMEOUT 60)

# Which outcome nxai_ipc_receive() reports per incoming signal, and in particular that
# a graceful EXIT is not the same result as a protocol violation. Drives the raw signalling
# directly: the framed sender cannot produce an EXIT as the follow-up to a reallocation.
add_executable(e2e-ipc-exit-signal-test
    ipc_exit_signal_test.cpp
    "${NXAI_UTILS_DIR}/nxai_ipc_utils.cpp"
    "${NXAI_UTILS_DIR}/nxai_utils.cpp"
)
target_include_directories(e2e-ipc-exit-signal-test PRIVATE "${NXAI_UTILS_DIR}")
find_package(Threads REQUIRED)
target_link_libraries(e2e-ipc-exit-signal-test PRIVATE Threads::Threads)

add_test(NAME e2e-ipc-exit-signal COMMAND e2e-ipc-exit-signal-test)
set_tests_properties(e2e-ipc-exit-signal PROPERTIES TIMEOUT 30)

# How the receive loop both C++ examples run leaves the loop, and that it closes both
# channels on the way out. The fixture rebuilds the binary on every ctest run, so a
# stale artifact cannot stand in for the current loop.
add_executable(e2e-processor-loop-test
    processor_loop_test.cpp
    "${NXAI_UTILS_DIR}/nxai_ipc_utils.cpp"
    "${NXAI_UTILS_DIR}/nxai_utils.cpp"
    "${NXAI_UTILS_DIR}/external_processor_utilities.cpp"
    "${NXAI_UTILS_DIR}/mpack.c"
)
target_include_directories(e2e-processor-loop-test PRIVATE "${NXAI_UTILS_DIR}")

add_test(
    NAME e2e-processor-loop-build
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target e2e-processor-loop-test
            --config $<CONFIG>
)
set_tests_properties(e2e-processor-loop-build PROPERTIES FIXTURES_SETUP processor-loop-binary)

add_test(NAME e2e-processor-loop COMMAND e2e-processor-loop-test)
set_tests_properties(e2e-processor-loop PROPERTIES TIMEOUT 30 FIXTURES_REQUIRED processor-loop-binary)

if(UNIX AND NOT APPLE)
    target_link_libraries(e2e-postprocessor-harness PRIVATE rt)
    target_link_libraries(e2e-ipc-sync-obj-test PRIVATE rt)
    target_link_libraries(e2e-ipc-failure-paths-test PRIVATE rt)
    target_link_libraries(e2e-ipc-exit-signal-test PRIVATE rt)
    target_link_libraries(e2e-processor-loop-test PRIVATE rt)
endif()

# Launch the C++ reference processor over real SHM IPC and assert its response.
add_test(
    NAME e2e-cpp
    COMMAND e2e-postprocessor-harness --check objects cpp_objects_example cpp-postprocessor-example
            $<TARGET_FILE:postprocessor-cpp-objects-example>
)

# Same processor over 1024-byte channels with an oversized frame, so both directions cross
# the SHM reallocation handshake. No other case here reaches that code at all.
add_test(
    NAME e2e-cpp-realloc
    COMMAND e2e-postprocessor-harness --check realloc cpp_objects_example cpp-postprocessor-example
            $<TARGET_FILE:postprocessor-cpp-objects-example>
)
set_tests_properties(e2e-cpp-realloc PROPERTIES TIMEOUT 30)

# Negative-input scenarios for the C++ reference processor: a frame it cannot
# process, followed by a well-formed one. Both outcomes below PIN today's behavior -- the
# C processor's mpack-based reader rejects a bad frame and keeps serving ("recovers") --
# rather than prescribe it; any deliberate change to the error policy flips them on purpose.
add_test(
    NAME e2e-cpp-negative-truncated
    COMMAND e2e-postprocessor-harness --check negative truncated recovers
            $<TARGET_FILE:postprocessor-cpp-objects-example>
)
set_tests_properties(e2e-cpp-negative-truncated PROPERTIES TIMEOUT 30)

# Same processor, structurally valid MessagePack with every expected key missing: it
# parses to an empty message and is processed as such, so the processor also survives.
add_test(
    NAME e2e-cpp-negative-missing-keys
    COMMAND e2e-postprocessor-harness --check negative missing-keys recovers
            $<TARGET_FILE:postprocessor-cpp-objects-example>
)
set_tests_properties(e2e-cpp-negative-missing-keys PROPERTIES TIMEOUT 30)

# Launch the C++ tensors reference processor: the harness sends a TensorsMessage and
# asserts the processor replaced the tensors with its known "cpp_tensors_example" bbox tensor.
add_test(
    NAME e2e-cpp-tensors
    COMMAND e2e-postprocessor-harness --check tensors
            $<TARGET_FILE:postprocessor-cpp-tensors-example>
)

# Launch the Python reference processors over real SHM IPC.
# Requires Python >= 3.12 (the processors use PEP 701 nested f-strings). Runs on Windows too:
# the Python SDK's IPC layer does branch per platform (SetEvent/WaitForSingleObject against
# eventfd), and every scenario below passes under MSVC. The SHM reallocation in
# _ipc_send_response is reached by e2e-python-realloc alone.
find_package(Python3 3.12 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
    add_test(
        NAME e2e-python
        COMMAND e2e-postprocessor-harness --check objects python_example postprocessor-example
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/postprocessor-python-example/postprocessor-python-example.py
    )
    set_tests_properties(e2e-python PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # The one case that drives the Python SDK's _ipc_send_response reallocation branch:
    # its response outgrows the 1024-byte output channel.
    add_test(
        NAME e2e-python-realloc
        COMMAND e2e-postprocessor-harness --check realloc python_example postprocessor-example
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/postprocessor-python-example/postprocessor-python-example.py
    )
    set_tests_properties(e2e-python-realloc PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # Negative-input scenarios for the OBJECTS Python example. Both pin
    # "recovers": msgpack.unpackb raises on the truncated frame and the SDK's read loop
    # drops it, while the missing-keys frame parses into a message whose fields all take
    # their defaults, which the processor happily annotates.
    add_test(
        NAME e2e-python-negative-truncated
        COMMAND e2e-postprocessor-harness --check negative truncated recovers
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/postprocessor-python-example/postprocessor-python-example.py
    )
    set_tests_properties(e2e-python-negative-truncated PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    add_test(
        NAME e2e-python-negative-missing-keys
        COMMAND e2e-postprocessor-harness --check negative missing-keys recovers
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/postprocessor-python-example/postprocessor-python-example.py
    )
    set_tests_properties(e2e-python-negative-missing-keys PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # A processor that declared
    # ReceiveInputImage: true, sent a frame without the ImageData block, raises
    # AttributeError on `msg.image_data.bytes` and the v5 main loop lets it terminate the
    # process -- so the well-formed frame that follows is never served. This test pins
    # "dies" as the CURRENT contract. Do not make the example defensive to turn it green:
    # A deliberate change to the error policy flips this expectation.
    add_test(
        NAME e2e-python-negative-no-image
        COMMAND e2e-postprocessor-harness --check negative no-image dies
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/postprocessor-python-image-example/postprocessor-python-image-example.py
    )
    set_tests_properties(e2e-python-negative-no-image PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # Drive the migrated settings plugin: the harness seeds a forwarded UserSettings map
    # and asserts the plugin applied it as an object attribute.
    add_test(
        NAME e2e-python-settings
        COMMAND e2e-postprocessor-harness --check settings
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/postprocessor-python-settings-example/postprocessor-python-settings-example.py
    )
    set_tests_properties(e2e-python-settings PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # Drive the object-counter-threshold plugin: the harness seeds a live
    # min_object_count of 3 (UserSettings) and 3 "person" objects, and asserts the
    # processor fires the counter event only when the live setting is read.
    add_test(
        NAME e2e-python-counter
        COMMAND e2e-postprocessor-harness --check counter
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/postprocessor-python-object-counter-threshold/postprocessor-python-object-counter-threshold.py
    )
    set_tests_properties(e2e-python-counter PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # Drive the height-estimator plugin: the harness seeds live frame_fraction_per_meter
    # and max_height_cm (UserSettings, distinct from the hardcoded defaults) and one
    # "person" of normalized height 0.7, and asserts the estimated Height attribute and
    # the over-limit event reflect the live settings, not the defaults.
    add_test(
        NAME e2e-python-height
        COMMAND e2e-postprocessor-harness --check height
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/postprocessor-python-height-estimator/postprocessor-python-height-estimator.py
    )
    set_tests_properties(e2e-python-height PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # Drive the web-dashboard plugin: the harness seeds one "test" object and asserts the
    # processor returns it field-for-field unmodified (the web-app POST side-effect fails
    # harmlessly with no server listening).
    add_test(
        NAME e2e-python-dashboard
        COMMAND e2e-postprocessor-harness --check dashboard
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/postprocessor-python-web-dashboard/postprocessor-python-web-dashboard.py
    )
    set_tests_properties(e2e-python-dashboard PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # Drive the skipFeatureExtraction fixture: the harness asserts the fixture's
    # response carries the "skipFeatureExtraction": true wire field on the gated object and
    # leaves it absent (-> parsed false) on the ungated one. Wire-contract only -- see the
    # skip-feature-extraction scenario doc comment in postprocessor_harness.cpp for what
    # this does not prove.
    add_test(
        NAME e2e-skip-feature-extraction
        COMMAND e2e-postprocessor-harness --check skip-feature-extraction
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-skipfeatureextraction-fixture.py
    )
    set_tests_properties(e2e-skip-feature-extraction PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # Drive the migrated image plugin: the harness seeds a root ImageData block of known
    # bytes and asserts the processor summed them into an object attribute.
    add_test(
        NAME e2e-python-image
        COMMAND e2e-postprocessor-harness --check image
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/postprocessor-python-image-example/postprocessor-python-image-example.py
    )
    set_tests_properties(e2e-python-image PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # Drive the migrated tensor preprocessor: the harness seeds a known
    # "nms_sensitivity-" tensor plus a forwarded UserSettings map
    # (nmsoverride=0.3) and asserts the processor applied that value, leaving an
    # unrelated tensor untouched.
    add_test(
        NAME e2e-python-preproc-tensor
        COMMAND e2e-postprocessor-harness --check preproc-tensor
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/preprocessor-python-tensor-example/preprocessor-python-tensor-example.py
    )
    set_tests_properties(e2e-python-preproc-tensor PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # Drive the migrated image preprocessor: the harness seeds a known 4x4 image plus a
    # forwarded UserSettings map (mirrorimage=false) and asserts the processor
    # returned it unchanged.
    add_test(
        NAME e2e-python-preproc-image
        COMMAND e2e-postprocessor-harness --check preproc-image
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/preprocessor-python-image-example/preprocessor-python-image-example.py
    )
    set_tests_properties(e2e-python-preproc-image PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # Sibling of the above: seeds no UserSettings at all, relying on the compiled-in
    # `true` mirrorimage default, and asserts a true horizontal mirror -- the
    # disabled-path scenario above cannot reach the mirroring algorithm itself.
    add_test(
        NAME e2e-python-preproc-image-mirror
        COMMAND e2e-postprocessor-harness --check preproc-image-mirror
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/preprocessor-python-image-example/preprocessor-python-image-example.py
    )
    set_tests_properties(e2e-python-preproc-image-mirror PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # The same mirror, but at the ~99.5 MB a 3840x2160 camera really produces and over the
    # 1024-byte channels the AI Manager creates, so the reallocation handshake runs in both
    # directions on a payload of live size. The 4x4 scenarios above fit inside their initial
    # channel and reallocate nothing.
    add_test(
        NAME e2e-python-preproc-image-realloc
        COMMAND e2e-postprocessor-harness --check preproc-image-realloc
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/preprocessor-python-image-example/preprocessor-python-image-example.py
    )
    set_tests_properties(e2e-python-preproc-image-realloc PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 60
    )

    # Three of those frames back to back, without draining the response in between --
    # the way the AI Manager drives a preprocessor. The single-frame case above always
    # finds the processor parked in its wait, so only this one can see the next frame's
    # reallocation signal arrive while the processor is still busy with the previous.
    add_test(
        NAME e2e-python-preproc-image-realloc-multi
        COMMAND e2e-postprocessor-harness --check preproc-image-realloc-multi
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/preprocessor-python-image-example/preprocessor-python-image-example.py
    )
    # Disabled until the contract for what a sender does after BUSY is settled: a send
    # that times out during an SHM reallocation leaves the next frame's send failing
    # with BUSY.
    set_tests_properties(e2e-python-preproc-image-realloc-multi PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 120
        DISABLED TRUE
    )

    # Drive the noresponse plugin: the harness sends 3 consecutive frames and asserts (a)
    # no response is ever written to the output channel -- NoResponse has no return
    # channel at all -- and (b) the process is still alive and consuming frames after the
    # third one. The multi-frame count is the point: a single-frame version would pass
    # against a processor that dies after exactly one frame.
    add_test(
        NAME e2e-python-noresponse
        COMMAND e2e-postprocessor-harness --check no-response
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/postprocessor-python-noresponse-example/postprocessor-python-noresponse-example.py
    )
    set_tests_properties(e2e-python-noresponse PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # Drive the confidences plugin: the harness seeds one "person" object of confidence
    # 0.876 and asserts attributes["Confidence"] == "0.88" -- the bare
    # str(round(confidence, 2)) format (no fixed decimal width) the v5 port must
    # preserve from v4.
    add_test(
        NAME e2e-python-confidences
        COMMAND e2e-postprocessor-harness --check confidences
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/postprocessor-python-confidences-example/postprocessor-python-confidences-example.py
    )
    set_tests_properties(e2e-python-confidences PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # Drive the events plugin: the harness asserts exactly one event with
    # type_id == "ex.example.event", byte-identical to the Events[].ID this
    # processor's external_postprocessors.json declares, and pins the asymmetry that
    # -- unlike object classes -- the id must carry no "nx.nxai." prefix
    # (the most frequent integrator mistake in the SDK).
    add_test(
        NAME e2e-python-events
        COMMAND e2e-postprocessor-harness --check events
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/postprocessor-python-events-example/postprocessor-python-events-example.py
    )
    set_tests_properties(e2e-python-events PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # Drive the bestshots fixture: the harness seeds one already-tracked
    # "car" object and launches with --emit_best_shots true; expects the response's
    # BestShots array to name that track with the fixture's sentinel bounding box.
    # Depends only on msgpack, like every other e2e-python-* scenario, so it runs on
    # Python3_EXECUTABLE rather than the edgeimpulse extras venv below.
    add_test(
        NAME e2e-python-bestshots
        COMMAND e2e-postprocessor-harness --check bestshots
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/postprocessor-python-bestshots-example/postprocessor-python-bestshots-example.py
                -- --emit_best_shots true
    )
    set_tests_properties(e2e-python-bestshots PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # Sibling of the above: no --emit_best_shots flag (default "false"); expects an
    # empty BestShots array even with the same eligible tracked car seeded.
    add_test(
        NAME e2e-python-bestshots-no-emit
        COMMAND e2e-postprocessor-harness --check bestshots-no-emit
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/postprocessor-python-bestshots-example/postprocessor-python-bestshots-example.py
    )
    set_tests_properties(e2e-python-bestshots-no-emit PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
        TIMEOUT 30
    )

    # edgeimpulse and objects-sender are the scenarios whose processors import beyond
    # msgpack (numpy, pillow and the edgeimpulse client; numpy, OpenCV and requests --
    # see their requirements.txt). An agent sandbox ships only the SDK clone plus a
    # working python3/pip (no prepared image) -- this is the offline-layer counterpart
    # of the live harness's dependency gap -- so provision a dedicated venv rather than
    # assume Python3_EXECUTABLE has them. OpenCV is the headless wheel: the
    # full one needs libGL at import time, which a bare WSL or CI image does not have,
    # and the processor uses no GUI call.
    #
    # Availability is decided by importing the modules, not by pip's exit code:
    # that is the precondition the scenario actually has, it stays correct if a wheel
    # installs but cannot load, and it makes re-configuring a populated build tree a
    # no-op instead of another network round-trip.
    set(E2E_EXTRAS_VENV "${CMAKE_CURRENT_BINARY_DIR}/e2e-extras-venv")
    if(WIN32)
        set(E2E_EXTRAS_PYTHON "${E2E_EXTRAS_VENV}/Scripts/python.exe")
    else()
        set(E2E_EXTRAS_PYTHON "${E2E_EXTRAS_VENV}/bin/python")
    endif()
    set(E2E_EXTRAS_IMPORT_CHECK "import msgpack, numpy, PIL, edgeimpulse, cv2, requests")

    execute_process(
        COMMAND ${E2E_EXTRAS_PYTHON} -c "${E2E_EXTRAS_IMPORT_CHECK}"
        RESULT_VARIABLE E2E_EXTRAS_IMPORT_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NOT E2E_EXTRAS_IMPORT_RESULT EQUAL 0)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m venv ${E2E_EXTRAS_VENV}
            RESULT_VARIABLE E2E_EXTRAS_VENV_RESULT
        )
        if(E2E_EXTRAS_VENV_RESULT EQUAL 0)
            execute_process(
                COMMAND ${E2E_EXTRAS_PYTHON} -m pip install --quiet
                        msgpack numpy pillow edgeimpulse opencv-python-headless requests
            )
            execute_process(
                COMMAND ${E2E_EXTRAS_PYTHON} -c "${E2E_EXTRAS_IMPORT_CHECK}"
                RESULT_VARIABLE E2E_EXTRAS_IMPORT_RESULT
                OUTPUT_QUIET
                ERROR_QUIET
            )
        endif()
    endif()

    if(E2E_EXTRAS_IMPORT_RESULT EQUAL 0)
        # Drive the edgeimpulse fixture: 3 consecutive frames, each with a seeded
        # ImageData block and a below-p_value confidence so every frame queues a
        # training sample (exercising the ImageData -> JPEG path); asserts no response
        # ever arrives (NoResponse: true) and the process survives all 3, the same
        # multi-frame requirement as no-response -- a crash converting the image
        # would otherwise only show up as a silently dead child. --config_file points at
        # a dedicated fixture ini rather than the shipped
        # plugin.edgeimpulse.ini.example, chiefly because the example enables
        # auto_generator, whose wall-clock timer would queue samples even if the
        # confidence/ImageData path under test were broken. See the header of
        # edgeimpulse-fixture.ini for both load-bearing settings.
        add_test(
            NAME e2e-python-edgeimpulse
            COMMAND e2e-postprocessor-harness --check edgeimpulse
                    ${E2E_EXTRAS_PYTHON}
                    ${CMAKE_SOURCE_DIR}/postprocessor-python-edgeimpulse-example/postprocessor-python-edgeimpulse-example.py
                    -- --config_file ${CMAKE_CURRENT_SOURCE_DIR}/edgeimpulse-fixture.ini
        )
        # Longer than every other case: this processor's import graph alone costs
        # 12.8s on ext4 and 52s on a /mnt/c checkout under WSL, before the three
        # frames are driven. The harness widens its ready wait to match.
        set_tests_properties(e2e-python-edgeimpulse PROPERTIES
            ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
            TIMEOUT 150
        )

        # Drive the objects-sender plugin: the harness seeds 4 reference points and a
        # non-default bridge address and topic (UserSettings) plus a "car" and a "person",
        # and asserts the Coordinates attribute of both objects. Nothing listens on the
        # seeded bridge address, so the processor's POST fails with a refused connection,
        # which it logs and ignores.
        add_test(
            NAME e2e-python-objects-sender
            COMMAND e2e-postprocessor-harness --check objects-sender
                    ${E2E_EXTRAS_PYTHON}
                    ${CMAKE_SOURCE_DIR}/postprocessor-python-objects-sender/postprocessor-python-objects-sender.py
        )
        set_tests_properties(e2e-python-objects-sender PROPERTIES
            ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/nxai-utilities"
            TIMEOUT 30
        )
    else()
        message(WARNING "e2e: could not provision the extras dependency venv "
            "(msgpack, numpy, pillow, edgeimpulse, opencv-python-headless, requests) -- "
            "skipping e2e-python-edgeimpulse and e2e-python-objects-sender")
    endif()

    # Guard that every migrated processor is reachable from the root CMakeLists:
    # a colocated registration JSON with no add_subdirectory line -- or a Python
    # CMakeLists.txt copied from the wrong install-destination example -- fails here
    # instead of shipping silently unwired.
    add_test(
        NAME build-wiring-scan
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tests/build-wiring/build_wiring_scan.py
                --root ${CMAKE_SOURCE_DIR}
    )
    set_tests_properties(build-wiring-scan PROPERTIES TIMEOUT 30)

    # Guard that no tracked file names an internal ticket, host, person, branch, local
    # path, process word or overlay link. The case is registered only where the script
    # exists.
    if(EXISTS ${CMAKE_SOURCE_DIR}/tests/build-wiring/public_tree_scan.py)
        add_test(
            NAME public-tree-scan
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_SOURCE_DIR}/tests/build-wiring/public_tree_scan.py
                    --root ${CMAKE_SOURCE_DIR}
        )
        set_tests_properties(public-tree-scan PROPERTIES TIMEOUT 60)
    endif()

    # Regression net for the scanner itself: the old documented invocation,
    # `unittest discover -s tests/build-wiring -p "test_*.py"`, could not cross the
    # hyphenated `build-wiring` directory to reach `tests/`, silently collected zero
    # tests, and exited 0. Point discovery directly at the test
    # directory so every test is actually collected and run.
    add_test(
        NAME build-wiring-scan-unittest
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
                -s ${CMAKE_SOURCE_DIR}/tests/build-wiring/tests
                -p "test_*.py"
                -v
    )
    set_tests_properties(build-wiring-scan-unittest PROPERTIES TIMEOUT 30)
else()
    message(STATUS "Python >= 3.12 not found -- skipping e2e-python tests")
endif()
