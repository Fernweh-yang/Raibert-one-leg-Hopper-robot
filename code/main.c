#include "variable.h"

char path[] = "../myproject/one_leg_robot/";
char xmlfile[] = "oneleg.xml";

//***************************
//used to render 
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}

void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}

void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}

void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}
//***************************


//***************************
// used to support control
void set_torque(const mjModel* m,int actuator_number,int flag)
{
  if (flag==0)
    m->actuator_gainprm[10*actuator_number+0]=0;
  else
    m->actuator_gainprm[10*actuator_number+0]=1;
}

void set_position(const mjModel* m,int actuator_number,double kp)
{
  m->actuator_gainprm[10*actuator_number+0]=kp;
  m->actuator_biasprm[10*actuator_number+1]=-kp;
}

void set_velocity(const mjModel* m,int actuator_number,double kv)
{
  m->actuator_gainprm[10*actuator_number+0]=kv;
  m->actuator_biasprm[10*actuator_number+2]=-kv;
}
//***************************


//***************************
// Controller
void init_controller(const mjModel* m, mjData* d)
{
  int actuator_number;

  actuator_number = 0; //hip_p
  set_position(m,actuator_number,100);
  actuator_number = 1; //hip_v
  set_velocity(m,actuator_number,10);

  actuator_number = 2; //knee_p
  set_position(m,actuator_number,1000);
  actuator_number = 3; //knee_v
  set_velocity(m,actuator_number,0);

  stateMachine = LOADING;
  step_number = 0;

}

void update_controller(double pos_foot[3], double z_velocity){
  if (stateMachine==LOADING && pos_foot[2]<=0.05)
  {
    stateMachine = COMPRESSION;
    //printf("COMPRESSION \n");
  }

  if (stateMachine==COMPRESSION && z_velocity>0)
  {
    stateMachine = THRUST;
    //printf("THRUST \n");
  }

  if (stateMachine==THRUST && pos_foot[2]>=0.05)
  {
    stateMachine = UNLOADING ;
    //printf("UNLOADING  \n");
  }

  if (stateMachine==UNLOADING  && z_velocity<0)
  {
    stateMachine = LOADING;
    //printf("LOADING \n");
    step_number = step_number+1;
    printf("%d \n",step_number);
  }
}

void controller(const mjModel* m, mjData* d)
{
  int body_number = 3;
  double pos_foot[3]={ d->xpos[3*body_number+0], d->xpos[3*body_number+1], d->xpos[3*body_number+2]};
  double z_velocity = d->qvel[1]; //0 is x, 1 is z

  update_controller(pos_foot,z_velocity);

  //all actions
  int actuator_number;
  if (stateMachine == LOADING)
  {
    actuator_number = 2; //knee_p
    set_position(m,actuator_number,1000);
    actuator_number = 3; //knee_v
    set_velocity(m,actuator_number,100);
  }

  if (stateMachine == COMPRESSION)
  {
    actuator_number = 2; //knee_p
    set_position(m,actuator_number,1000);
    actuator_number = 3; //knee_v
    set_velocity(m,actuator_number,0);


  }

  if (stateMachine == THRUST)
  {
    actuator_number = 2; //knee_p
    set_position(m,actuator_number,1050);
    actuator_number = 3; //knee_v
    set_velocity(m,actuator_number,0);

    d->ctrl[0] = -0.2;
  }

  if (stateMachine == UNLOADING )
  { 
    actuator_number = 2; //knee_p
    set_position(m,actuator_number,1000);
    actuator_number = 3; //knee_v
    set_velocity(m,actuator_number,100);

    d->ctrl[0] = 0;
  }
}
//***************************


//***************************
// main
int main(int argc, const char** argv)
{

    // activate software
    mj_activate("mjkey.txt");
    char xmlpath[100]={};
    char datapath[100]={};
    strcat(xmlpath,path);
    strcat(xmlpath,xmlfile);

    // load and compile model
    char error[1000] = "Could not load binary model";

    // check command-line arguments
    if( argc<2 )
        m = mj_loadXML(xmlpath, 0, error, 1000);

    else
        if( strlen(argv[1])>4 && !strcmp(argv[1]+strlen(argv[1])-4, ".mjb") )
            m = mj_loadModel(argv[1], 0);
        else
            m = mj_loadXML(argv[1], 0, error, 1000);
    if( !m )
        mju_error_s("Load model error: %s", error);

    // get data of model
    d = mj_makeData(m);


    // init GLFW
    if( !glfwInit() )
        mju_error("Could not initialize GLFW");
    // create window
    GLFWwindow* window = glfwCreateWindow(1244, 700, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    // initial visual data structure
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);                
    mjr_makeContext(m, &con, mjFONTSCALE_150);   
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);
    // set the camera view
    double arr_view[] = {90, -10, 5, 0.000000, 0.000000, 2.000000};
    cam.azimuth = arr_view[0];
    cam.elevation = arr_view[1];
    cam.distance = arr_view[2];
    cam.lookat[0] = arr_view[3];
    cam.lookat[1] = arr_view[4];
    cam.lookat[2] = arr_view[5];

    // add Control
    mjcb_control = controller;
    init_controller(m,d);

    // Simulate
    while( !glfwWindowShouldClose(window))
    {
        mjtNum simstart = d->time;
        while( d->time - simstart < 1.0/60.0 )
        {
            mj_step(m, d);
        }

        if (d->time>=simulation_endtime)
        {
           break;
         }

       // view at body
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
        int body_number;
        body_number = 1;
        cam.lookat[0] = d->xpos[3*body_number+0];
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);
        glfwSwapBuffers(window);
        glfwPollEvents();

    }

    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();
    return 1;
}
//***************************
