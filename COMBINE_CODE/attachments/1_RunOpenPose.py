import glob 
import os
import subprocess


fileslist = glob.glob('*.mov')
print('Running OpenPose on the following .MOV files in current directory (make sure that OpenPose models folder is in the current directory)):\n', fileslist, '\n')

OpenPoseEXEPath = 'C:\\Users\\khur4\\Documents\\openpose-1.6.0-binaries-win64-gpu-flir-3d_recommended\\openpose\\bin\\OpenPoseDemo.exe' 

file_txt = open('file_ids.txt','w+')

for file in fileslist:
    os.mkdir('file_'+str(fileslist.index(file))+'_videoframes')
    os.mkdir('file_'+str(fileslist.index(file))+'_JSON')
    file_txt.write('file_'+str(fileslist.index(file))+' = '+str(file)+'\n')
    #print('Running: \n',OpenPoseEXEPath, '--video', file, '--write_images', 'file_'+str(fileslist.index(file))+'_videoframes', '--write_json', 'file_'+str(fileslist.index(file))+'_JSON','\n')
    subprocess.run([OpenPoseEXEPath, '--video', file, '--write_images', 'file_'+str(fileslist.index(file))+'_videoframes', '--write_images_format','jpg','--write_json', 'file_'+str(fileslist.index(file))+'_JSON', '\n'])
    #subprocess.run([OpenPoseEXEPath, '--video', file, '--write_images', 'file_'+str(fileslist.index(file))+'_videoframes', '\n'])

file_txt.close()