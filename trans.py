import os
from loguru import logger
import pandas as pd

import video_WCT as WCT
#global set
cuda_index=0
basePath=os.path.join(os.getenv('HOME'),"Pictures")
#basePath="/home/jiefangjun/Pictures/"
stylesPath="styles"


logger.add('trans_logger.txt')
originalPaths=["LDV2_train240","LDV2_val15","LDV2_test15"]
processedPaths=["LDV2_train240_WCT","LDV2_val15_WCT","LDV2_test15_WCT"]
alphas=[0.6,0.7,0.8,0.9,1.0]

#load data frame.
train_df=pd.read_excel('data_train.xlsx',sheet_name='Sheet1',engine='openpyxl',usecols='A:E',nrows=240)
val_df=pd.read_excel('data_val.xlsx',sheet_name='Sheet1',engine='openpyxl',usecols='A:E',nrows=15)
test_df=pd.read_excel('data_test.xlsx',sheet_name='Sheet1',engine='openpyxl',usecols='A:E',nrows=15)
#set video format
train_df['Video']=train_df['Video'].apply(lambda x:format(x,'03d'))
val_df['Video']=val_df['Video'].apply(lambda x:format(x,'03d'))
test_df['Video']=test_df['Video'].apply(lambda x:format(x,'03d'))
#set index
train_df.index=train_df['Video']
val_df.index=val_df['Video']
test_df.index=test_df['Video']

df=[train_df,val_df,test_df]

#styles
styles=[file for file in os.listdir(os.path.join(basePath,stylesPath)) \
    if file.endswith('.jpg')]
styles.sort()

if not os.path.exists('taskfile.txt'):
    taskfile=open('taskfile.txt','w')
    for task_index in range(1):
        originalPath=originalPaths[task_index]
        processedPath=processedPaths[task_index]
        videos=os.listdir(os.path.join(basePath,originalPath))
        videos.sort()
        for style in styles:
            style_name=style.split('.')[0]
            fullstyle=os.path.join(basePath,stylesPath,style)
            for alpha in alphas:
                for video in videos:
                    frame_rate=df[task_index].loc[video,'Frame rate']
                    fullPro=os.path.join(basePath,processedPath,style_name,f"alpha{alpha}",f'{video}.mp4')
                    fullOri=os.path.join(basePath,originalPath,video)
                    taskfile.write(f'{fullOri} {fullstyle} {fullPro} {frame_rate} {alpha}\n')
    taskfile.close()

taskfile=open('taskfile.txt','r')
all_tasks=taskfile.readlines()
taskfile.close()
all_tasks=[t.split() for t in all_tasks]

if not os.path.exists('done_taskfile.txt'):
    done_taskfile=open('done_taskfile.txt','w')
    done_taskfile.close()
    
done_taskfile=open('done_taskfile.txt','r+')
done_tasks=done_taskfile.readlines()
done_tasks=[t.split() for t in done_tasks]

tasks=[t for t in all_tasks if t not in done_tasks]

done_n=len(done_tasks)
all_n=len(all_tasks)
logger.info(f"progress bar:{done_n}/{all_n}")

for ori,style,pro,frame_rate,alpha in tasks:
    try:
        # create processed dir
        pro_dir=os.path.dirname(pro)
        if not os.path.exists(pro_dir):
            os.makedirs(pro_dir)
        
        logger.info(f"Start to process {pro}")
        WCT.main(ori,style,pro,float(alpha),int(frame_rate),cuda=True,gpu=cuda_index)
        # os.system(f"python video_WCT.py  \
        #     --contentPath {ori} --stylePath {style}  \
        #         --output {pro} --alpha {alpha} --fps {frame_rate} --cuda --gpu {cuda_index}")
        logger.info(f"{pro} processed successfully!\n")
        done_taskfile=open('done_taskfile.txt','a+')
        done_taskfile.write(f"{ori} {style} {pro} {frame_rate} {alpha}\n")
        #logger.error(f"{pro} processed failed!\n")
    except Exception as e:
        logger.error(f"{pro} processed failed!\n{e}")