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
        print ('You need to install boto3. Try pip install --upgrade boto3')
    
    try:
        Experiment.create(experiment_name=experiment_name, 
                            description="Running AutoGluon Tabular with SageMaker Experiments", 
                            sagemaker_boto_client=client)
    except:
        print ('Could not create the experiment. Is your basename properly configured?')
    
    return experiment_name