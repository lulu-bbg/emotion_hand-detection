# emotion_hand-detection
This code connect face and hand gestures in images for easier detection of the emotion they represent.

![Example image for Happy](https://github.com/lulu-bbg/emotion_hand-detection/blob/main/happy_hand%20(2).jpg?raw=true)

## The Algorithm
We used a convolutional neural network (CNN) called resnet18 which basically expidites how nano computers undersatnd patterns in different images of the same thing, this could be a lot of different things like species, actions, emotions, foods, etc. as long as it looks different and you can give it a name. 

There are three main component sizes that need to be specificed when running CNN: batch, workers, and epochs. 
    1. The batch size will determine how many images are shown to the machine at once. 
    2. Quantity of workers decides how many different batches are proccessed by the machine at once. 
    3. The epochs are how many times this entire process will happen.
      - Each epoch shows you Train Loss, Train Accuracy, Val Loss, and Val Accuracy.
          * Train Loss: A value that guides to the optimization process during training, which basically assesses the error of the model on the training set.
          * Train Accuracy: Percentage of correct predictions over the dataset.
          * Val Loss: A value showing how much information is lost when the program reverse engineers the images (compares the output to the input) which assess the performance of a deep learning model on the validation set.
          * Val Accuracy: Percentage of correct predictions over random images.
Explanations from baeldung.
    
There are some limitations with how this CNN works since it is the programmers job to find out how many epochs are over or under fit. This is because running the same images too many times will create certain niches that are far too specific to fit all the more generalizing patterns in the images, but too little will create too little connections and therfore will misclasify images that should not be under the same class.
How to find the goldilocks ratio for your own program is pretty loose and on a testing basis. The ideal is set by the limitations of your own equipment. 
However anything under ten to thirty epchos is pretty unreliable as seen on this data table:
![Accuracy- Efficiency chart for Epochs](https://github.com/lulu-bbg/emotion_hand-detection/blob/main/Accuracy.png?raw=true)
![Loss- Efficiency chart for Epochs](https://github.com/lulu-bbg/emotion_hand-detection/blob/main/Loss.png?raw=true)

## Running this project

1. Chose the emotions you want to represent. Each should have a different facial and hand gesture.
2. Take at least 150 images for each emotion and label each based on the emotion and a number. Ex mad1, mad2, mad3, etc.
3. On visual studio code go to jetson-inference/python/training/classification/models and make a folder labeled emotions.
4. Inside the emotions folder make three folders labeled train, test, and val alongside a file named labels.txt.
5. labels.txt should have an alphabetical list with the emotions chosen.
6. The train, test, and val folders should each have a folder named after each emotion. Ex mad, sad, happy, etc.
7. These emotion folders should each have 10% of the images for their designated emotion in test and val, while the other 80% is in train. (This can be done through FileZilla.)
8. Then go back to jetson-inference and into ./docker/run.sh, this will allow you to start a new container, execute a command the docker, and pull an image if needed. (which we do need)
9. Inside the docker go to python/training/classification and input: python3 train.py --model-dir=models/emotions data/emotions (This is the Python script that contains the code for training the model.)
    - You can play around with the how the model runs with: --batch-size=NumberOfBatchFiles --workers=NumberOfWorkers --epochs=NumberOfEpochs
    - Then, 'Ctl+C' allows you to stop at anytime, while '--resume' allows you to train your modle where you left off instead of restarting.
11. When the program is done running through the epochs, ensure you are in jetson-inference/python/training/classification in the docker to then input python3 onnx_export.py --model-dir=models/emotions. This will save the training as resnet18.onnx to your emotions folder.
12. 'Ctl+D' will exit docker so we can go to our nano now and see what the training did, which also helps us find possible adjustments
    - Go into jetson-inference/python/training/classification
    - Set 'NET=models/emotiuons' and 'DATASET=data/emotions' (This simplifies the paths into NET and DATASET)
    - Finally run, imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/emotions/'image input name'.jpg 'image output name'.jpg
      or
      imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt /dev/video0 (if no monitor+) demo.mp4

## The Result
Trail One- Low percentages of emotion detection(23-30%). Angry, sad, and none were detected easily, but suprised and happy were undetectable. This led to adding hand gestures to the images for the training process to have more distinct patterns between the images.

Trail Two- Video Demo: https://youtu.be/ixWe0OLodj4
(Nano is detecting all five emotion, but needs different angles to recognize the patterns.)

For 100 epochs and 100 train images, this trail went pretty well. However, while taking the images it is likley I adjusted the camera, even if seeminly insignificant, this had a pretty big impact on what emotions the nano could detect in slightly different angles. Image detection emotions:

!(https://github.com/lulu-bbg/emotion_hand-detection/blob/main/sup.jpg?raw=true)
!(https://github.com/lulu-bbg/emotion_hand-detection/blob/main/sad.jpg?raw=true)
!()
!()
!()
