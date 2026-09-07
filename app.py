import tempfile
import cv2
import streamlit as st
from ultralytics import YOLO

st.title("License Plate Recognition System")
st.write(
    "Upload a car video to detect license plates, extract text, and view the"
    " annotated output."
)

uploaded_file = st.file_uploader(
    "Choose a video...", type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
  tfile = tempfile.NamedTemporaryFile(delete=False)
  tfile.write(uploaded_file.read())
  input_path = tfile.name

  st.video(uploaded_file)

  if st.button("Process Video"):
    with st.spinner("Processing frames... Please wait."):
      cap = cv2.VideoCapture(input_path)
      # use a temporary file for output to prevent browser/Streamlit caching
      out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
      output_path = out_tfile.name

      fourcc = cv2.VideoWriter_fourcc(*"VP80")

      fps = cap.get(cv2.CAP_PROP_FPS) or 30
      width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

      out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

      model = YOLO("license_plate_best.pt")

      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break

        results = model(frame, verbose=False)
        for r in results:
          for box in r.boxes:
            conf = float(box.conf.item())
            if conf < 0.3:
              continue
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        out.write(frame)

      cap.release()
      out.release()

    st.success("Processing complete!")
    st.video(output_path)