#!/usr/bin/env python

"""Find Orbbec cameras and print their serial numbers/names for use with lerobot."""

import pyorbbecsdk as ob

def find_orbbec_cameras():
    context = ob.Context()
    device_list = context.query_devices()
    count = device_list.get_count()

    if count == 0:
        print("No Orbbec devices detected.")
        return

    print(f"Found {count} Orbbec device(s):\n")
    for i in range(count):
        dev = device_list[i]
        info = dev.get_device_info()
        name = info.name() if hasattr(info, "name") else "Orbbec"
        serial = info.serial_number() if hasattr(info, "serial_number") else ""
        
        print(f"Device {i}:")
        print(f"  Name:   '{name}'")
        print(f"  Serial: '{serial}'")
        print(f"\nTo use this camera, run:")
        if serial:
            print(f"  python examples/test_depth_camera.py --type orbbec --serial_number_or_name {serial} --width 640 --height 480 --fps 30")
        else:
            print(f"  python examples/test_depth_camera.py --type orbbec --serial_number_or_name '{name}' --width 640 --height 480 --fps 30")
        print()

if __name__ == "__main__":
    find_orbbec_cameras()
