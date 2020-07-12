# Get CIFAR10

DATASET_NAME='CIFAR-10'
DATASET_DIR_NAME='cifar10'
DATASET_PATH='http://filecremers3.informatik.tu-muenchen.de/~dl4cv/cifar10_train.zip'
DATASET_ZIP_NAME='cifar10_train.zip'
EXERCISE_DIR=$(pwd)

cd ..
mkdir -p datasets/$DATASET_DIR_NAME
cd datasets/$DATASET_DIR_NAME
DATASET_DIR=$(pwd)

echo 'Downloading '$DATASET_NAME
curl -LO $DATASET_PATH
tar -xzvf $DATASET_ZIP_NAME
rm $DATASET_ZIP_NAME
echo $DATASET_NAME ' downloaded successfully!!! '

cd $EXERCISE_DIR
