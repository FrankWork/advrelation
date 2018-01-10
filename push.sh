#! /bin/bash

# server=liuzhihui@219.216.64.88:/home/liuzhihui/advrelation/
# server=lzh@219.216.77.206:/home/lzh/work/python/qa-rc/advrelation/
server=renfeiliang@219.216.64.90:/home/renfeiliang/lzh/advrelation/

rsync -avh --delete --exclude=saved_models/  \
           --exclude=__pycache__/   \
           --exclude=.vscode/   \
           --exclude=embed300.google.npy \
           --exclude=google_embed300.npy.tar.gz \
           --exclude=google_words.lst \
           --exclude=data/generated/ \
           --exclude=data \
      ./ $server