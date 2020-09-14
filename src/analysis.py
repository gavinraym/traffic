
   
    # This links target classes to sign names (i.e. stop sign is [class 14] = ['Stop'])
    class_map = pd.read_csv('data/meta/meta_map.csv', index_col=0)
   
    # We will record all pertinent information into a dictionary which 
    # will be saved as a json for use by the website
    report = dict()
    report['title'] = {model_name}
            
    # Add basic stats and scores of the model to the report
    acc_preds = preds[preds['predicted_class']==preds['actual_class']]
    report['True Predictions'] = len(acc_preds)
    report['Total predictions'] = len(preds)
    report['Accuracy score'] = len(acc_preds)/len(preds)

    # The next section of the report looks at the test data separated into class
    for sign in range(43):

        # Each class's data is saved as a nested dictionary
        class_dict = dict()

        # Basic class data is recorded
        class_dict['number'] = sign
        class_dict['name'] = class_map.iloc[sign][0]
        class_dict[f'Accuracy Score'] = group_preds.loc[sign][0] / len(sign_preds)        

        # Prediction dfs are prepared for this particular class
        sign_preds = preds[preds.actual_class == sign]
        group_preds = sign_preds.groupby('predicted_class').count()



        # Class specific predictions are recorded as well. This includes the number
        # of each class that was predicted across this class's test images
        class_dict['preds'] = pr

            # Record results of predictions for this class
            class_dict[num] = group_preds
        


        # Class specific data dicts are added to the models report dict
        report[sign] = class_dict   

        

    # Saves the report
    with open(f'{path}/report.pk', 'wb') as pck:
        pickle.dump(report, pck) 

    # Samples of truthily classified images are saved for review
    os.mkdir(f'{path}/true_imgs')
    
    samples = true_imgs.sample(min(len(true_imgs), 100))
    for img_loc in samples.image_location:
        shutil.copyfile(f'data/test/{img_loc}', f'{path}/true_imgs/{img_loc}')

    # Samples of falsely classified images are saved for review
    os.mkdir(f'{path}/false_imgs')
    false_imgs = sign_preds[sign_preds.actual_class != sign_preds.predicted_class]
    samples = false_imgs.sample(min(len(true_imgs), 100))
    for img_loc in samples.image_location:
        shutil.copyfile(f'data/test/{img_loc}', f'{path}/false_imgs/{img_loc}')