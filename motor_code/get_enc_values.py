# import can
# import struct

# def read_encoder_values(bus):
#     # CAN ID for encodr values (in decimal)
#     encoder_can_id = 0x009  # This is 9 in decimal
#     encoder_can_id_1 = 0x029
#     encoder_can_id_2 = 0x049
#     encoder_can_id_3 = 0x069
#     encoder_can_id_4 = 0x089
#     encoder_can_id_5 = 0x0A9
#     encoder_can_id_6 = 0x0C9
#     encoder_can_id_7 = 0x0E9
    
    
#     ## turn the motor on
#     # Send a message with the CAN ID and the data
#     msg_0 = can.Message(arbitration_id=(0x07), data=struct.pack('<I', 8), is_extended_id = False)
#     msg_1 = can.Message(arbitration_id=(0x27), data=struct.pack('<I', 8), is_extended_id = False)
#     msg_2 = can.Message(arbitration_id=(0x47), data=struct.pack('<I', 8), is_extended_id = False)
#     msg_3 = can.Message(arbitration_id=(0x67), data=struct.pack('<I', 8), is_extended_id = False)
#     msg_4 = can.Message(arbitration_id=(0x87), data=struct.pack('<I', 8), is_extended_id = False)
#     msg_5 = can.Message(arbitration_id=(0xA7), data=struct.pack('<I', 8), is_extended_id = False)
#     msg_6 = can.Message(arbitration_id=(0xC7), data=struct.pack('<I', 8), is_extended_id = False)
#     msg_7 = can.Message(arbitration_id=(0xE7), data=struct.pack('<I', 8), is_extended_id = False)

#     bus.send(msg_0)
#     bus.send(msg_1)
#     bus.send(msg_2)
#     bus.send(msg_3)
#     bus.send(msg_4)
#     bus.send(msg_5)
#     bus.send(msg_6)
#     bus.send(msg_7)


# # Initialize the CAN interface
# bus = can.interface.Bus(channel='can0', bustype='socketcan')

# # Read encoder values
# encoder_value = read_encoder_values(bus)
# print(f"Final Encoder Value: {encoder_value}")



# import can
# import struct

# def read_all_encoder_values_once(bus):
#     # CAN IDs for encoder values (in decimal)
#     encoder_can_ids = [0x009, 0x029, 0x049, 0x069, 0x089, 0x0A9, 0x0C9, 0x0E9]
#     # encoder_can_id = 

#     # CAN IDs for turning on motors
#     motor_on_ids = [0x07, 0x27, 0x47, 0x67, 0x87, 0xA7, 0xC7, 0xE7]

#     # # Send turn-on commands for each motor
#     for motor_id in motor_on_ids:
#         msg = can.Message(arbitration_id=motor_id, data=struct.pack('<I', 8), is_extended_id=False)
#         bus.send(msg)
    
#     print("Turn-on commands sent. Listening for encoder values...")
    
    # for motor_id in motor_on_ids:
    #     msg = can.Message(arbitration_id=motor_id, data=struct.pack('<I', 1), is_extended_id=False)
    #     bus.send(msg)
    
    # print("Turn-off commands sent. Not Listening for encoder values...")

#     # Dictionary to store encoder values
#     encoder_values = {}

#     # Listen for encoder values, exit once all are received
#     while len(encoder_values) < len(encoder_can_ids)+1:
#         msg = bus.recv(timeout=1)  # Timeout to prevent infinite waiting
#         if msg is None:
#             print("Timeout: No more messages received.")
#             break

#         # if msg.arbitration_id in encoder_can_ids:
#             # # Unpack the position value (assuming 32-bit float)
#             # encoder_value = struct.unpack('<f', msg.data[:4])[0]
#             # encoder_values[msg.arbitration_id] = encoder_value
#             # msg = can.Message(arbitration_id=motor_id, data=struct.pack('<I', 1), is_extended_id=False)
#             # bus.send(msg)
#             # print(f"Encoder Value Received - CAN ID {hex(msg.arbitration_id)}: {encoder_value}")
#         if msg.arbitration_id in encoder_can_ids:
#             encoder_value = struct.unpack('<f', msg.data[:4])[0]
#             encoder_values[msg.arbitration_id] = encoder_value
            
#             # Turn off the corresponding motor
#             motor_index = encoder_can_ids.index(msg.arbitration_id)
#             motor_id = motor_on_ids[motor_index]  # Find the correct motor ID
#             off_msg = can.Message(arbitration_id=motor_id, data=struct.pack('<I', 0), is_extended_id=False)
#             bus.send(off_msg)
    
#     print(f"Encoder Value Received - CAN ID {hex(msg.arbitration_id)}: {encoder_value}")
#     print(f"Motor turned off - CAN ID {hex(motor_id)}")


#     return encoder_values

# Initialize the CAN interface
# bus = can.interface.Bus(channel='can0', bustype='socketcan')

# # Read encoder values once
# encoder_values = read_all_encoder_values_once(bus)

# Print all encoder values
# print("\nFinal Encoder Values:")
# for can_id, value in encoder_values.items():
#     print(f"CAN ID {hex(can_id)}: {value}")


import can
import struct
import time

def read_all_encoder_values_once(bus):
    # encoder_can_ids = [0x009, 0x029, 0x049, 0x069, 0x089, 0x0A9, 0x0C9, 0x0E9]
    # motor_on_ids = [0x07, 0x27, 0x47, 0x67, 0x87, 0xA7, 0xC7, 0xE7]
    
    encoder_can_ids = [0x029, 0x049, 0x0A9, 0x0C9]
    motor_on_ids = [0x27, 0x47, 0xA7, 0xC7]

    # Turn on all motors with a slight delay to prevent collisions
    for motor_id in motor_on_ids:
        msg = can.Message(arbitration_id=motor_id, data=struct.pack('<I', 8), is_extended_id=False)
        bus.send(msg)
        time.sleep(1)  # Prevent overload

    print("Turn-on commands sent. Listening for encoder values...")

    time.sleep(5)
    encoder_values = {}
    while len(encoder_values) < len(encoder_can_ids):
        msg = bus.recv(timeout=1)
        if msg is None:
            print("Timeout: No more messages received.")
            break

        if msg.arbitration_id in encoder_can_ids:
            encoder_value = struct.unpack('<f', msg.data[:4])[0]
            encoder_values[msg.arbitration_id] = encoder_value

            # Turn off motor after reading value
            motor_index = encoder_can_ids.index(msg.arbitration_id)
            motor_id = motor_on_ids[motor_index]
            off_msg = can.Message(arbitration_id=motor_id, data=struct.pack('<I', 1), is_extended_id=False)
            bus.send(off_msg)

            print(f"Encoder Value Received - CAN ID {hex(msg.arbitration_id)}: {encoder_value}")
            print(f"Motor turned off - CAN ID {hex(motor_id)}")

    return encoder_values
    
    # print("set input and condtrol mode to 3")
    # bus.send(can.Message(arbitration_id=(0x2B), data=[0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00], is_extended_id = False))
    # bus.send(can.Message(arbitration_id=(0x4B), data=[0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00], is_extended_id = False))
    # bus.send(can.Message(arbitration_id=(0xAB), data=[0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00], is_extended_id = False))
    # bus.send(can.Message(arbitration_id=(0xCB), data=[0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00], is_extended_id = False))
    
    # Motor_1_base = -2.9374332427978516
    # Motor_2_base = -0.3438396453857422
    # Motor_5_base = -2.3755226135253906
    # Motor_6_base = -6.191577911376953
    
    
    # print("setting negative 0.5 from home position")
    # bus.send(can.Message(arbitration_id=(0x2D), data=struct.pack('<ff', Motor_1_base, 0.0), is_extended_id = False))
    # bus.send(can.Message(arbitration_id=(0x4D), data=struct.pack('<ff', Motor_2_base, 0.0), is_extended_id = False))
    # bus.send(can.Message(arbitration_id=(0xAD), data=struct.pack('<ff', Motor_5_base, 0.0), is_extended_id = False))
    # bus.send(can.Message(arbitration_id=(0xCD), data=struct.pack('<ff', -5.865492820739746, 0.0), is_extended_id = False))


# Initialize the CAN interface
bus = can.interface.Bus(channel='can0', bustype='socketcan')

encoder_values = read_all_encoder_values_once(bus)

print("\nFinal Encoder Values:")
for can_id, value in encoder_values.items():
    print(f"CAN ID {hex(can_id)}: {value}")

