import time
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas

def display_frame(device, bitmap):
    with canvas(device) as draw:
        for y, row in enumerate(bitmap):
            for x, pixel in enumerate(row):
                if pixel:
                    draw.point((x, y), fill="white")

def heart_animation():
    serial = spi(port=0, device=0, gpio=noop())
    device = max7219(serial, cascaded=1)
    device.contrast(16)
    
    print("Heart animation... Press Ctrl+C to stop")
    
    # Define the 7x7 heart outline shape
    heart_shape = [
        [0,1,1,0,1,1,0],
        [1,0,0,1,0,0,1],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [0,1,0,0,0,1,0],
        [0,0,1,0,1,0,0],
        [0,0,0,1,0,0,0],
    ]
    
    # Transpose of the heart shape
    heart_shape_transpose = [
        [0,1,1,1,0,0,0],
        [1,0,0,0,1,0,0],
        [1,0,0,0,0,1,0],
        [0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0],
        [1,0,0,0,1,0,0],
        [0,1,1,1,0,0,0],
    ]
    
    # Create empty 7x7 frame
    empty_frame = [[0 for _ in range(7)] for _ in range(7)]
    
    try:
        while True:
            # Phase 1: Heart appears bit by bit from bottom to top
            current_frame = [row[:] for row in empty_frame]  # Deep copy
            
            # Light up row by row from bottom (row 6) to top (row 0)
            for row in range(6, -1, -1):
                # Copy the heart row to current frame
                for col in range(7):
                    if heart_shape[row][col]:
                        current_frame[row][col] = 1
                
                display_frame(device, current_frame)
                time.sleep(0.2)  # Delay between each row lighting up
            
            # Small pause before blinking starts
            time.sleep(0.5)
            
            # Phase 2: Heart blinks twice
            for blink in range(2):
                # Turn off (show empty frame)
                display_frame(device, empty_frame)
                time.sleep(0.3)
                
                # Turn on (show complete heart)
                display_frame(device, heart_shape)
                time.sleep(0.3)
            
            # Pause before repeating the whole sequence
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        # Clear the display when stopping
        with canvas(device) as draw:
            pass
        print("\nAnimation stopped")

if __name__ == "__main__":
    heart_animation()
