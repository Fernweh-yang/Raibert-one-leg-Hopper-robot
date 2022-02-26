#include<stdbool.h> //for bool
#include<unistd.h> //for usleep
#include <math.h>
#include "mujoco.h"
#include "glfw3.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

double simulation_endtime = 20;
int step_number;

typedef enum{
  LOADING       = 0X00, // Zero hip torque
  COMPRESSION   = 0X01, // Body attitude with hip
  THRUST        = 0X02, // Spring give a thrust + Body attitude with hip
  UNLOADING     = 0X03, // Zero hip torque
  //FLIGHT        = 0X04, // position leg for loading
}stateMachineTypeDef;
stateMachineTypeDef stateMachine;

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

// holders of one step history of time and position to calculate dertivatives
mjtNum position_history = 0;
mjtNum previous_time = 0;

// controller related variables
float_t ctrl_update_freq = 100;
mjtNum last_update = 0.0;
mjtNum ctrl;
