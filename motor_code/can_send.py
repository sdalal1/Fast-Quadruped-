import can
import struct

# Initialize the CAN bus
bus = can.interface.Bus("can0", bustype="socketcan")

try:
    # Send messages
    bus.send(can.Message(arbitration_id=0x27, data=struct.pack('<I', 8), is_extended_id=False))
    bus.send(can.Message(arbitration_id=0x47, data=struct.pack('<I', 8), is_extended_id=False))
except can.CanError as e:
	print("failed closed state")
try:
    bus.send(can.Message(arbitration_id=0x2B, data=[0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00], is_extended_id=False))
    bus.send(can.Message(arbitration_id=0x4B, data=[0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00], is_extended_id=False))
except can.CanError as e:
	print("Cant set input and control")

try:
    bus.send(can.Message(arbitration_id=0x4D, data=struct.pack('<ff', 1.0, 0.0), is_extended_id=False))
    bus.send(can.Message(arbitration_id=0x2D, data=struct.pack('<ff', 1.0, 0.0), is_extended_id=False))
except can.CanError as e:
	print("cant set position")

try:
    bus.send(can.Message(arbitration_id=0x27, data=struct.pack('<I', 1), is_extended_id=False))
    bus.send(can.Message(arbitration_id=0x47, data=struct.pack('<I', 1), is_extended_id=False))

    print("All CAN messages sent successfully!")
except can.CanError as e:
    print(f"Failed to send a CAN message: {e}")
except KeyboardInterrupt:
    print("Script interrupted.")
