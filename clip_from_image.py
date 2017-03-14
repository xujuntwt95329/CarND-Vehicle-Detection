# create a video clip from bunch of image
from moviepy.editor import ImageSequenceClip

clip = ImageSequenceClip('test', fps=3)
clip.write_videofile('crowdai.mp4', audio=False)
