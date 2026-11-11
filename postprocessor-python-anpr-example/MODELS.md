# ANPR models

Two ONNX models drive the ANPR postprocessor. Both ship in the repo's
`example_models/` directory at the root.

## Models included

| File | Role |
|------|------|
| `example_models/License_Plate_Detector.onnx` | Detects license plate bounding boxes |
| `example_models/License_Plate_OCR.onnx` | Reads character logits from the detected plate region |

## Uploading to AI Manager

Custom models are uploaded through the NX AI cloud portal.

1. Log in to the NX AI cloud portal for your customization (for example,
   NX Meta VMS). This is the portal where you assign a pipeline to a device.
   The address of the portal depends on the customization.
2. Open the model upload page and follow the upload wizard.
3. Upload both model files. The wizard auto-assigns the correct normalization
   values for these models (mean `[0, 0, 0]`, std `[1, 1, 1]`); no manual
   entry is required.

## Chaining the models

In the AI Manager pipeline editor, configure the pipeline as follows:

1. Add **License_Plate_Detector** as the primary (detector) model.
2. Chain **License_Plate_OCR** in **Feature Extraction** mode.
3. Set the extracted feature name to **`LP`**. This is a pipeline-level label
   and is unrelated to `ocr_output_name` in the registration JSON: the engine
   writes the runtime's own tensor names into the `Tensors` block verbatim, which for this model is `Identity:0`.
4. Select the **ANPR-Example** postprocessor for the **License_Plate_OCR**
   model only. Each of its messages carries one plate crop's character logits
   plus the plate's track ID (`OriginalObjectID`), which is all the
   postprocessor needs to set the plate's text on the detector's object. Leave
   the detector model without a postprocessor.
