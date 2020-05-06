# Simulating The Solar System

As decsribed in the report to go alongside this software, the scripts in this repository aim to solve the N-body problem.

Each folder operates independantly of the others.
The optimal software to use depends on the use-case (see the report).
To run the programs ensure the csv file is within the same folder as the program and referenced correctly in the script.


DATA FILES
The data filesshould be in CSV format. They should be in order of distance from center of the system and should have the parameters:        "NAME", x position, y position, Mass, x velocity, y velocity
for each body.

2-BODY PROBLEM
Only functions on systems that have 2 bodies and one body is central. This can be changed by no longer fixing body one's accelerationa and position.

N-BODY PROBLEM
The N-body solution functions and a reasonable time step for the example of the Solar System is around 12,000.

ADAPTIVE TIME STEP
An adaptive time step was created in order to reduce accuracy and increase efficency. The parameters for this can be canged dependiong on your needs. For reasonable values of the Solar System the decreasing time step parameters should be around 0.99 and 1.01 and the increasing parameters should be around 0.9999 and 1.0001. This should allow for reasonable time steps to be achieved.
