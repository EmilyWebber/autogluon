from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent
from smexperiments.tracker import Tracker

import boto3
import time

def set_experiment_config(experiment_basename = None):
    '''
    Optionally takes an base name for the experiment. Has a hard dependency on boto3 installation. 
    Creates a new experiment using the basename, otherwise simply uses autogluon as basename.
    May run into issues on Experiments' requirements for basename config downstream.
    '''
    now = int(time.time())
    
    if experiment_basename:
        experiment_name = '{}-autogluon-{}'.format(experiment_basename, now)
    else:
        experiment_name = 'autogluon-{}'.format(now)
    
    try:
        client = boto3.Session().client('sagemaker')
    except:
        print ('You need to install boto3 to create an experiment. Try pip install --upgrade boto3')
        return ''
    
    try:
        Experiment.create(experiment_name=experiment_name, 
                            description="Running AutoGluon Tabular with SageMaker Experiments", 
                            sagemaker_boto_client=client)
        print ('Created an experiment named {}, you should be able to see this in SageMaker Studio right now.'.format(experiment_name))
        
    except:
        print ('Could not create the experiment. Is your basename properly configured? Also try installing the sagemaker experiments SDK with pip install sagemaker-experiments.')
        return ''
    
    return experiment_name

def create_trial(experiment_name, trial_base_name = None):
    '''
    Requires a valid experiment name, optionally takes a base trial name.
        Attempts to create a trial for a new model associated with the pre-created experiment.
    '''
    now = int(time.time())
    
    if trial_base_name:
        trial_name = "autogluon-{}-{}".format(trial_base_name, now)    
    else:
        trial_name = "autogluon-candidate-{}".format(now)
    
    try:
        client = boto3.Session().client('sagemaker')
    except:
        print ('You need to install boto3. Try pip install --upgrade boto3')
        return ''
    
    try:
        trial = Trial.create(trial_name=trial_name, 
                            experiment_name=experiment_name,
                            sagemaker_boto_client=client)
        print ('Created a trial named {}'.format(trial_name))
        return trial
        
    except:
        print ('Could not create a trial, was that a valid experiment or trial base name?')
        
    return ''