# My-Curiosity-Rover

For the project work have been taken as a source of inspiration the recent NASA space missions Curiosity and Perseverance which have as their primary objective the search for traces of water within Martian soil and rock samples.
The project, in fact, is a two-dimensional simulation of a rover whose primary objective is to explore and analyze r rocks distributed in a portion of Martian territory m × n with total area A.
It is assumed that the rover has mapped a series of rocks in a Martian plain and autonomously goes to analyze them one by one to find traces of water.
Once in the proximity of a rock the rover analyzes it and if it finds traces of water it collects the sample and continue on its way to the next rock. 
Once finished the analysis of all the rocks, the rover must bring the samples to its base located in the center of the map at coordinates (0, 0).

### Importing libraries
At first, the software must import all the Python libraries needed for the simulation in order to run:

* **random**, for random data generation;
  

* **numpy**, it allows to work with vectors and matrices in a more efficient and fast way than it is possible to do with lists and lists of lists (matrices); moreover, one of the strong points of numpy is to be able to work on vectors exploiting the optimizations of vector calculation of the processor of the machine;


* **matplotlib**, allows you to plot 2D graphs in various ways, allows you to plot numpy ndar- rays, but also lists that are transformed directly into arrays when passed as plot arguments;


* **math**, allows you to extend the mathematical functionality of Python.

### Generation of rocks to explore
The **_generate_rocks()_** function generates the number of rocks to be explored by the rover.
It takes as input the num_rocks argument used to generate the coordinates of the individual rocks included in the coord_x and coord_y lists declared at the beginning of the function.
The for loop inside is the heart of the function: this is where the coordinates of the rocks with **rnd.randint(-num_rocks, num_rocks)** are randomly generated and inserted inside the lists coord_x and coord_y; moreover, a flag is randomly generated, through the same function used for the coordinates, which is inserted in the list **water_traces** and which can take values 0 or 1 to indicate if a certain rock has traces of water (1) or not (0). In output returns the generated lists.

### Generating rover goals
The **_rover_goals()_** function takes as input the coord_x, coord_y and water_traces lists generated by the **_generate_rocks()_** function. 
The purpose of this function is to generate the ordered sequence of targets that the rover must visit while exploring the Martian territory.
To do this, the distances of individual rocks from the base of the rover are calculated according to the **Euclidean Distance** heuristic
and are inserted into the distances list declared at the beginning of the function. 
Next, the distance relative to the individual rock and its coordinates are inserted into the tutti list.
Finally, the all_goals list is sorted in ascending order and its elements are inserted into the goals list which is returned as the output of the function.

### Movement functions of the rover
In order to reach the targets on the two-dimensional map, the rover will have to perform a series of movements:

* **_go_ahead(start_x=0, start_y=0, final_y=0, pose=[])_**, takes as input the coordinates of the position where the rover is, the yi coordinate it has to reach and the list of rover positions.
  This function calculates in a for loop the various positions that the rover takes during the route and inserts them inside the list that will then be returned in output;
    

* **_go_back(start_x=0, start_y=0, final_y=0, pose=[])_**, takes as input the coordinates of the position where the rover is, the yi coordinate it has to reach and the list of the rover positions.
  This function calculates in a for loop the various positions that the rover takes during the route and inserts them inside the list that will then be returned in output;
  

* **_turn_right(start_x=0, start_y=0, final_x=0, pose=[])_**, takes as input the coordinates of the position where the rover is, the xi coordinate it has to reach and the list of the rover positions.
  This function calculates in a for loop the various positions the rover takes on the way and inserts them within the list that will then be returned in output;
  

* **_turn_left(start_x=0, start_y=0, final_x=0, pose=[])_**, takes as input the coordinates of the position where the rover is, the xi coordinate it has to reach and the list of the rover positions.
  This function calculates in a for loop the various positions the rover takes on the way and inserts them within the list that will then be returned in output;
  

* **_hold_position(start_x=0, start_y=0)_**, takes as input the coordinates of the position where the rover is.
  This function returns in output the position where the rover is located.
  
### Rover movement algorithm
The **_move_to_goal()_** function is the heart of the implementation of the rover movement on the two-dimensional map as it contains the path planning algorithm for the movement which is based on the fulfillment of several conditions based on the objectives to be achieved and the positions assumed by the rover during the route.
The function takes as input the list of goals and positions the rover is in during the route.
It first evaluates whether a target in the list contains water traces or not and then, based on the relative position of the target with respect to the position where the rover is, it evaluates which movements to make by calling the appropriate functions previously described.
In output it returns the list of positions to be taken by the rover.

### Function to return to base
Once all targets have been explored, rocks analyzed, and useful samples collected, the rover must return to base to deposit the samples.
To do this, the function **_back_to_base()_** is used, which takes in input the list of rover positions and updates it by adding the new positions to return from the position of the last target, i.e. the last rock visited, to the base.
In output it returns the updated position list.

### Generating the rover path
The **_rover_path()_** function generates the path to be followed by the rover on the two-dimensional map.
It takes in input the list of targets to be reached by the rover and sequentially executes the functions **_move_to_goal()_** to generate the list of positions from the base to the rocks and the function **_back_to_base()_** to update the list of positions by adding those needed to return from the last target, i.e. the last visited rock, to the base.
In output returns the complete list of positions.

### Map generation
The **_rocks_map()_** function generates the two-dimensional map of the Martian soil with markers for rocks without water, rocks with water traces, and the rover placed on it.
The terrain is created thanks to the function **_generate_map()_** which returns two-dimensional matrices of X, Y and Z points; the latter is populated by distributed data with Normal distribution having mean 0 and variance 1.
These data are used to produce two-dimensional Digital Elevation Models, in this way graphically simulating the Martian terrain with its elevations and gorges.
The functions init() and animate() are used respectively to initialize and animate the simulation which is saved in a GIF file.

### Simulation function
Finally, the **_simulation()_** function is the main function that starts the simulation once it is called.
It defines at the beginning the number of rocks randomly generated in a range [10, 20] thanks to the function **rnd.randint(10, 20)** of the random library.
The function generates in order: the coordinates of the rocks, the goals of the rover, the movements the rover has to make on the map and finally the map.

