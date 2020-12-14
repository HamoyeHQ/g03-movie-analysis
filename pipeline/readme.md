A Docker is a set of platform service products that use OS-level virtualization to deliver software in packages called containers. The great advantage of Docker is that it makes your deployment easy and when the time comes you can even deploy it on multiple machines with almost no extra effort.


#Building pipelines through docker images

- To build pipeline through heavy-weight (reusable) containers, the functions need two modifications:

    - accept context string as a sys argv
    - write the result to file
    - no need to have subprocess installations inside the functions, as ‘requirements.txt’ used for building the docker image would take care of that
    
    
#Building Docker Images

- Docker images need to be created now in the usual manner with no changes to Dockerfile as the arguments to be passed at ENTRYPOINT will be taken care of by KFP.

#Uploading Created Images

- Once the images are build we need to upload the images to a location accessible by the pods in the cluster (i.e private or public docker registry).


#Stitiching the pipeline

- Once the images have been pushed we are ready to stitch the pipeline once again.
Unlike light-weight containers we don’t need to convert functions to container operations and directly proceed to stitch the pipeline together.


#importing KFP pipeline
from kfp.dsl import pipeline

#defining pipeline meta
@pipeline(
    name='Calculate Average',
    description='This pipeline calculates average'
)

#importing container operation
from kfp.dsl import Containerop

#stitch the steps
def average_calculation_pipeline(context: str=context):
    step_1 = ContainerOp(
        name = 'add', # name of the operation
        image = 'docker.io/avg/add', #docker location in registry
        arguments = [context], # passing context as argument
        file_outputs = {
            'context': '/output.txt' #name of the file with result 
        }
    )


   step_2 = ContainerOp(
        name = 'div', # name of operation   
        image = 'docker.io/avg/add', #docker location in registry
        arguments = [step_1], # passing step_1.output as argument
        file_outputs = {
            'context': '/output.txt' #name of the file with result 
        }
   )


#Compiling Kubeflow Pipeline
#importing KFP compiler
from kfp.compiler import Compiler
#compiling the created pipeline 
Compiler().compile(average_calculation_pipeline, 'pipeline.zip')


#Initialising Pipeline Run through Script
This zip file produced after the compilation can either be uploaded to create a kubeflow pipeline through the Kubeflow UI route or can be created using the following script.


#importing KFP client
from kfp import client
#initialising client instance
client = kfp.Client()
#creating experiment
experiment = client.create_experiment(
    name = "Average Experiment",
    description = "This is the average experiment"
)

#Define a run name 
run_name = "This is test run: 01"
#Submit a pipeline run
run_result = client.run_pipeline(
    experiment.id, 
    run_name,
    pipeline_filename,
    params = {}
)
print(run_result)



#Summary
#Below is a summary of the steps involved in creating and using a component:

- Write the program that contains your component’s logic. The program must use specific methods to pass data to and from the component.
- Containerize the program.
- Write a component specification in YAML format that describes the component for the Kubeflow Pipelines system.
- Use the Kubeflow Pipelines SDK to load and run the component in your pipeline.






























