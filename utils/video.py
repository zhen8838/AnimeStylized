import cv2
from pathlib import Path
from typing import Generator, Tuple


def get_read_stream(path: Path) -> Tuple[Generator, int, int, int, int]:
  read_stream = cv2.VideoCapture(path.as_posix())
  length = int(read_stream.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = int(read_stream.get(cv2.CAP_PROP_FPS))
  height = int(read_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
  width = int(read_stream.get(cv2.CAP_PROP_FRAME_WIDTH))

  def gen(stream):
    while True:
      ret, frame = stream.read()
      if ret:
        yield frame
      else:
        stream.release()
        break
  return gen(read_stream), length, fps, height, width


def get_writer_stream(path: Path, fps: int, height: int, width: int):
  writer_stream = cv2.VideoWriter(
      filename=path.as_posix(),
      fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
      fps=fps,
      frameSize=(width, height))
  return writer_stream
