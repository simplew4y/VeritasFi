import time
import json
import logging
import statistics
from functools import wraps

class Profiler:
    """A class to handle function profiling functionality"""
    
    def __init__(self):
        self.logger = logging.getLogger('function_profiler')
        self.profile_data = {}
        self._active_timers = {}
        self.metrics = {}
        
    def profile_function(self, func=None, name=None):
        """ Decorator to profile function calls and execution time
        Can be used as @profile_function or @profile_function(name="custom_name")
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed_time = time.perf_counter() - start_time
                
                # Use custom name if provided, otherwise use function name
                func_name = name if name is not None else func.__name__
                
                # Initialize tracking for this function if not already tracked
                if func_name not in self.profile_data:
                    self.profile_data[func_name] = {
                        'calls': 0,
                        'total_time': 0.0,
                        'execution_times': []
                    }
                    
                # Update statistics
                self.profile_data[func_name]['calls'] += 1
                self.profile_data[func_name]['total_time'] += elapsed_time
                self.profile_data[func_name]['execution_times'].append(elapsed_time)
                
                return result
            return wrapper
        
        # Handle both @profile_function and @profile_function(name="custom_name")
        if func is None:
            # Called as @profile_function(name="custom_name")
            return decorator
        else:
            # Called as @profile_function
            return decorator(func)
    
    def start(self, function_name):
        """Manually start timing a function or code block"""
        # Initialize tracking for this function if not already tracked
        if function_name not in self.profile_data:
            self.profile_data[function_name] = {
                'calls': 0,
                'total_time': 0.0,
                'execution_times': []
            }
        
        # Store start time
        self._active_timers[function_name] = time.perf_counter()
        return function_name  # Return name for convenience in end() calls
    
    def end(self, function_name):
        """Manually end timing a function or code block"""
        if function_name not in self._active_timers:
            # self.logger.warning(f"No active timer found for '{function_name}'. Did you call start() first?")
            return
        
        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._active_timers[function_name]
        
        # Update statistics
        self.profile_data[function_name]['calls'] += 1
        self.profile_data[function_name]['total_time'] += elapsed_time
        self.profile_data[function_name]['execution_times'].append(elapsed_time)
        
        # Clean up
        del self._active_timers[function_name]
        
        return elapsed_time  # Return elapsed time for convenience
    
    def add_metric(self, name, value):
        """Add a numeric metric to a list in the profiler under the given name"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
        return value  # Return value for convenience
    
    def log_profiling_results(self, output_file=None):
        """Log profiling results for all tracked functions and metrics"""
        all_stats = {}
        
        # Process function timing data
        for func_name, data in self.profile_data.items():
            if data['calls'] > 0:
                # Convert execution times to milliseconds
                execution_times_ms = [t * 1000 for t in data['execution_times']]
                
                # Calculate metrics
                count = data['calls']
                min_time = min(execution_times_ms) if execution_times_ms else 0
                max_time = max(execution_times_ms) if execution_times_ms else 0
                mean_time = statistics.mean(execution_times_ms) if execution_times_ms else 0
                
                # Calculate additional statistics if we have enough data
                std_dev = 0
                median = 0
                percentile_95 = 0
                percentile_99 = 0
                if len(execution_times_ms) > 0:
                    median = statistics.median(execution_times_ms)  # 50th percentile
                    if len(execution_times_ms) > 1:
                        std_dev = statistics.stdev(execution_times_ms)
                        # Calculate 95th and 99th percentiles
                        sorted_times = sorted(execution_times_ms)
                        idx_95 = int(0.95 * len(sorted_times))
                        idx_99 = int(0.99 * len(sorted_times))
                        percentile_95 = sorted_times[idx_95]
                        percentile_99 = sorted_times[idx_99]
                
                self.logger.info(f"Function '{func_name}' profiling results:")
                self.logger.info(f"  Total calls: {count}")
                self.logger.info(f"  Min execution time: {min_time:.4f} ms")
                self.logger.info(f"  Max execution time: {max_time:.4f} ms")
                self.logger.info(f"  Mean execution time: {mean_time:.4f} ms")
                self.logger.info(f"  Median execution time (50%): {median:.4f} ms")
                self.logger.info(f"  95th percentile: {percentile_95:.4f} ms")
                self.logger.info(f"  99th percentile: {percentile_99:.4f} ms")
                self.logger.info(f"  Std dev of execution time: {std_dev:.4f} ms")
                
                # Collect stats for this function
                all_stats[func_name] = {
                    'count': count,
                    'min': min_time,
                    'max': max_time,
                    'mean': mean_time,
                    'median': median,  # 50th percentile
                    'p95': percentile_95,
                    'p99': percentile_99,
                    'std_dev': std_dev
                }
        
        # Process custom metrics
        for metric_name, values in self.metrics.items():
            if values:
                # Calculate static metrics
                metric_stats = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'average': statistics.mean(values)
                }
                
                # Calculate median (50th percentile) and other percentiles
                if len(values) > 0:
                    sorted_values = sorted(values)
                    metric_stats['median'] = statistics.median(values)  # 50th percentile
                    
                    # Calculate 95th and 99th percentiles if there's enough data
                    if len(values) > 1:
                        idx_95 = int(0.95 * len(sorted_values))
                        idx_99 = int(0.99 * len(sorted_values))
                        metric_stats['p95'] = sorted_values[idx_95]
                        metric_stats['p99'] = sorted_values[idx_99]
                        metric_stats['std_dev'] = statistics.stdev(values)
                
                self.logger.info(f"Metric '{metric_name}' statistics:")
                self.logger.info(f"  Count: {metric_stats['count']}")
                self.logger.info(f"  Min: {metric_stats['min']}")
                self.logger.info(f"  Max: {metric_stats['max']}")
                self.logger.info(f"  Average: {metric_stats['average']:.4f}")
                self.logger.info(f"  Median (50%): {metric_stats.get('median', 0):.4f}")
                self.logger.info(f"  95th percentile: {metric_stats.get('p95', 0):.4f}")
                self.logger.info(f"  99th percentile: {metric_stats.get('p99', 0):.4f}")
                
                # Add to all stats
                all_stats[f"metric_{metric_name}"] = metric_stats
        
        # Save to file if requested
        if output_file and all_stats:
            with open(output_file, "w") as f:
                json.dump(all_stats, f, indent=2)
            self.logger.info(f"Profiling results saved to {output_file}")

    def reset(self):
        self.profile_data = {}
        self._active_timers = {}
        self.metrics = {}

# Create a global profiler instance
profiler = Profiler()
