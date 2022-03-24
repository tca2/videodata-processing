
# Choose random number between 1800 and 3600 (30th minute to 60th minute in video), to get the starting timestamp for the video, then then add 60 to that number to get ending timestamp for that video. 
# ffmpeg -ss [start] -i in.mp4 -t [duration] -c copy out.mp4
# For example ffmpeg -ss 1855 -i in.mp4 -t 60 -c copy out.mp4 will create out.mp4 file from 1855 to 1915 seconds
# shuf -i 1800-3600 -n 1

from ffmpy import FFmpeg
import os
from random import randrange
directory = os.listdir(input("Type the video directory (with the concatenated .mov files) from which you want to randomly sample 1 minute clips:"))

for video_file in directory:
	random_start_time = str(randrange(1800, 3540, 1))
	if video_file.endswith(".mov"):
		ff = FFmpeg(executable='C:\\Users\\khur4\\Documents\\ffmpeg-2021-04-04-git-b1b7cc698b-full_build\\bin\\ffmpeg.exe',\
			inputs={'data\\'+str(video_file): ['-ss',random_start_time]}, \
				outputs={video_file[18:-5]+'_1_minute_sample_from_'+random_start_time+'s'+'.mov': ['-t','60','-c','copy']})
		ff.run()



