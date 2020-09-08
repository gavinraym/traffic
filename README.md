# Capstone 3 proposal


## **Option One** - Keeping it simple:

In 2011, the German Traffic Sign Detection Benchmark competition was held, asking participants to classify pictures of road signs. I have included a pdf about it in this directory. For my capstone, I would like to design a CNN architecture pipeline that can classify these signs. I would like to use the dataset from this competition to train a CNN model that does better than the winners (98.98%).

The photos have already been altered in 3 ways: HOG, HAAR, and Hue hist. I will use these to make 4 different nueral network classifiers. These will make up the base layer of my pipeline. 

I will use the output from the base layer cnns to train a fifth cnn that classifies each photo based on which base layer cnn is most appropriate for it. The idea here is that the photos are taken in different conditions, and some are blurry or whatever. Each baselayer cnn will be potentially better for certain photo types. By classifying the photos into 4 categories (based on which base layer cnn is most accurate for that photo), and then training a fifth cnn to act as the beginning layer of the pipeline, I may be able to achieve higher overall accuracy.

The last step of my pipeline will be to add a 44th class (there are 43 casses of signs, and then I'll add one more for signs that the cnn is uncertain about.) My thought here is that self driving cars could make catestrophic (deadly) mistakes by mis-classifying signs. It would be much more advantageous to acknowledge when we are uncertain about what a sign is. This would allow the car to use other information sources, ask the user for help, or simply slow down and adjust the cameras to get a better look at the sign.

Potential difficulties: Achieving unbalanced classes in the base layer. To prevent this, I will plan on sorting the photos based on the outputted probabilities from the base layer cnns. Groupings will be made with the photos with highest correct probability for each cnn, with no one cnn getting more than 25% of the overall photos. I will evaluate the effectiveness of this model by calculating the overal effectiveness of each cnn compared to the others with all the data that is NOT in it's data set. Each base layer cnn should do worse on the data going to the other cnns than 1) it does on its own data 2) than the other cnns do on there own.

I really like this project idea because I am incredibly interested in self-driving cars. This is totally a direction I could go in for my career.

## **Option Two** - Not so simple:

Like any proud father, I take lots of videos of my son. Many of these go up on a Whatsapp thread for all his grandparents to see. My son is 2.5 yo, so a lot of the recent videos are showcasing his emerging speaking ability. 

It's all fine and dandy, except for one thing. Because I am holding the camera and therefore closer to the microphone, my voice is always much louder than his. Many of the best videos are ruined by the fact my prompting of him to say things is so very much louder than when he actually says it!

It would be great to have a filter that reduces (or even removes) my voice from a video. I think this would also make a great capstone three project.



