# ======================== IMPORTS ========================
from typing import List


# ====================== ENCODERS ========================
class Encoder:
    def gap_encode(self, numbers: List[int]) -> List[int]:
        if not numbers:
            return []
        
        gaps = [numbers[0]]
        for i in range(1, len(numbers)):
            gap = numbers[i] - numbers[i - 1]
            if gap < 0:
                raise ValueError("Input list must be sorted in ascending order.")
            gaps.append(gap)
        return gaps
    
    def gap_decode(self, gaps: List[int]) -> List[int]:
        if not gaps:
            return []
        
        numbers = [gaps[0]]
        for i in range(1, len(gaps)):
            number = numbers[-1] + gaps[i]
            numbers.append(number)
        return numbers
    
    def varbyte_encode(self, numbers: List[int]) -> bytes:
        stream = bytearray()
        for number in numbers:
            while True:
                byte = number & 127 # Get the last 7 bits
                number >>= 7        # Shift right by 7 bits
                if number == 0:
                    stream.append(byte | 128) # Set the stop bit to 1
                    break
                else:
                    stream.append(byte)       # Stop bit is 0
        return bytes(stream)
    
    def varbyte_decode(self, stream: bytes) -> List[int]:
        numbers = []
        number = 0
        shift = 0
        
        for byte in stream:
            if byte & 128: # Stop bit is 1
                number |= (byte & 127) << shift
                numbers.append(number)
                number = 0
                shift = 0
            else:          # Stop bit is 0
                number |= (byte & 127) << shift
                shift += 7
        return numbers
