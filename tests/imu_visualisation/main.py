import sys
sys.path.append("../..")

import json
import time
import LibsOpenGLGUI
import imu_debug
import json_serial_port_parser

serial_port_name = "/dev/tty.usbserial-141130"

serial_port = json_serial_port_parser.JsonSerialPortParser(serial_port_name, baudrate=9600)

'''
while True:
    serial_port.process()
    if serial_port.udpated():
        json_data = serial_port.get()
        print(json_data)
'''

'''
with open("./resources/result.json") as json_file:
    json_data = json.load(json_file)
'''

gui     = LibsOpenGLGUI.GLGui("./resources/gui.json")
debug   = imu_debug.IMUDebug()


while gui.main_step() != True:
    serial_port.process()
    if serial_port.udpated():
        json_data = serial_port.get()
        print("new data")
        print(json_data)
        debug.update(gui, json_data)
    else: 
        time.sleep(0.02)
        print("waiting for data")
