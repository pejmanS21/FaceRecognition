FILE=Face_Recognition_checkpoint_resnet50.pth
if [ ! -f "$FILE" ]; then
    echo "Downloading $FILE ..."
    # https://drive.google.com/file/d/1-sizR8uHD20KlL2thqVsEgXtDJE8cDpU/view?usp=sharing
    gdown --id 1-sizR8uHD20KlL2thqVsEgXtDJE8cDpU
else
    echo "$FILE Exist!"
    
fi
