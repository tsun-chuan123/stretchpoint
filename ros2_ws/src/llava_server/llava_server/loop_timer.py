#!/usr/bin/env python3

import time


class LoopTimer:
    """
    Simple loop timer to measure and display loop timing statistics.
    Based on the loop_timer.py from visual servoing examples.
    """
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.iteration_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        
    def start_of_iteration(self):
        """Mark the start of a loop iteration"""
        self.start_time = time.time()
        
    def end_of_iteration(self):
        """Mark the end of a loop iteration and update statistics"""
        self.end_time = time.time()
        
        if self.start_time is not None:
            iteration_time = self.end_time - self.start_time
            self.total_time += iteration_time
            self.iteration_count += 1
            
            if iteration_time < self.min_time:
                self.min_time = iteration_time
            if iteration_time > self.max_time:
                self.max_time = iteration_time
                
    def get_average_time(self):
        """Get average time per iteration"""
        if self.iteration_count > 0:
            return self.total_time / self.iteration_count
        return 0.0
        
    def get_frequency(self):
        """Get average frequency (Hz)"""
        avg_time = self.get_average_time()
        if avg_time > 0:
            return 1.0 / avg_time
        return 0.0
        
    def pretty_print(self):
        """Print formatted timing statistics"""
        if self.iteration_count > 0:
            avg_time = self.get_average_time()
            freq = self.get_frequency()
            print(f"Loop Timer Stats - Iterations: {self.iteration_count}, "
                  f"Avg: {avg_time*1000:.1f}ms ({freq:.1f}Hz), "
                  f"Min: {self.min_time*1000:.1f}ms, "
                  f"Max: {self.max_time*1000:.1f}ms")
        else:
            print("Loop Timer Stats - No iterations completed")
            
    def reset(self):
        """Reset all statistics"""
        self.iteration_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
