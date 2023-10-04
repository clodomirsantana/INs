# Interaction Networks (INs) #

<figure>
  <img src="example_ins.png" alt="example" width="100%"/>
  <figcaption><b>Fig. 1</b> - Example of interaction networks for the ABC, GA, and PSO. Nodes represent individuals in the population, and edges indicate that two nodes interacted (e.g. shared information)..</figcaption>
</figure> <br><br><br>

Interaction networks are a framework to model and study the behaviour of population-based metaheuristics. This repository provides a Python implementation and examples of models for 12 metaheuristics. Additionally, we included Jupyter notebooks with a few examples of comparisons and visualisations.

Additional information can be found at:

- Santana, Clodomir, Marcos Oliveira, Carmelo Bastos-Filho, and Ronaldo Menezes. ["Beyond exploitation: Measuring the impact of local search in swarm-based memetic algorithms through the interactions of individuals in the population." Swarm and Evolutionary Computation 70 (2022): 101040](https://www.sciencedirect.com/science/article/pii/S2210650222000128).
- Santana, Clodomir, Edward Keedwell, and Ronaldo Menezes. ["Networks of evolution: modelling and deconstructing genetic algorithms using dynamic networks." Proceedings of the Genetic and Evolutionary Computation Conference Companion. 2022.](https://dl.acm.org/doi/abs/10.1145/3520304.3529039?casa_token=-sxiX9JDXncAAAAA:jwXrdDWoidU5uItKpCHZNUyfPb7-42RPV_1sEATKo0KyXGjEBDtH0BCuEWaXIEcCW9cnvDy8L4rYPQ)

## How do I get set up? ###

* Before running the code please intall the required packages listed in the `requirements.txt` file.
* Examples of how to capture the interaction networks can be found in the folder `algorithms`.
* To generate the network files for the algorithms given, please execute the files named `RunAlgorithmName.py` in the folder `experiments`.
* The implementation of the interaction networks and other methods can be found in the folder `networks`.
* Inside the `experiments` folder is another folder named `notebooks` with examples of how to process and visualise the INs. Please note that the examples provided are just for illustrative purposes.

## Who do I talk to? ###

* Questions about the IN framework, metrics and other methods can he directed to Clodomir Santana (clodomir@ieee.org) or Prof Ronaldo Menezes (r.menezes@exeter.ac.uk)
* Although we tried out best to provide accurate implementation of the methods, if you find errors or something is not clear, please send a mesage to Clodomir Santana (clodomir@ieee.org)

## License

This work is free. You can redistribute it and/or modify it under the terms of the GNU Public license and subject to all prior terms and licenses imposed by the free, public data sources provided. The code comes without any warranty to the extent permitted by applicable law.
