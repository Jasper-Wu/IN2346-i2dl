
MODELS_DIR='models/*'
CODE_DIR='exercise_code/*py'
CLASSIFIERS_DIR='exercise_code/classifiers/*py'
NOTEBOOKS='*.ipynb'
DATASET_ZIP_NAME='exercise_1_submission.zip'
EXERCISE_DIR=$(pwd)

echo 'Zipping file '$DATASET_ZIP_NAME
zip -r $DATASET_ZIP_NAME $MODELS_DIR $CODE_DIR $NOTEBOOKS $CLASSIFIERS_DIR
echo $DATASET_ZIP_NAME ' created successfully!'
echo 'To submit your models upload the zip file to: https://dvl.in.tum.de/teaching/submission/'
