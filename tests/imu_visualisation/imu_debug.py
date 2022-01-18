import numpy

class IMUDebug:

    def __init__(self):
        
        self.record_length      = 100
        self.current_ptr        = 0 

        self.graph_roll_angle   = 180*(2.0*numpy.random.rand(self.record_length) - 1.0)
        self.graph_pitch_angle  = 180*(2.0*numpy.random.rand(self.record_length) - 1.0)
        self.graph_yaw_angle    = 180*(2.0*numpy.random.rand(self.record_length) - 1.0)

        self.x_axis             = numpy.array(range(self.record_length))

    def update(self, gui, json_data):

        angles = {}
        angles[0] =  float(json_data["imu_sensor"][0])
        angles[1] =  float(json_data["imu_sensor"][1])
        angles[2] =  float(json_data["imu_sensor"][2])

        gui.variables.add("imu_angles", angles)

        self.graph_roll_angle[self.current_ptr]     = angles[0]
        self.graph_pitch_angle[self.current_ptr]    = angles[1]
        self.graph_yaw_angle[self.current_ptr]      = angles[2]

        gui.variables.add("graph_roll_angle", [self.current_ptr, self.x_axis, self.graph_roll_angle])
        gui.variables.add("graph_pitch_angle", [self.current_ptr, self.x_axis, self.graph_pitch_angle])
        gui.variables.add("graph_yaw_angle",  [self.current_ptr, self.x_axis, self.graph_pitch_angle])

        self.current_ptr = (self.current_ptr+1)%self.record_length
