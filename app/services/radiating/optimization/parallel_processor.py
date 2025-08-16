"""
Parallel Processor

Concurrent entity processing with thread pool management, work queue optimization,
and load balancing for the radiating system.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Callable, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
from collections import deque, defaultdict
import threading
import queue
import multiprocessing

from app.core.radiating_settings_cache import get_radiating_settings
from app.core.redis_client import get_redis_client

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    """Worker states"""
    IDLE = "idle"
    BUSY = "busy"
    BLOCKED = "blocked"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class WorkItem:
    """Represents a unit of work"""
    id: str
    func: Callable
    args: Tuple
    kwargs: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 30.0


@dataclass
class WorkerMetrics:
    """Worker performance metrics"""
    worker_id: str
    state: WorkerState = WorkerState.IDLE
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    current_task: Optional[str] = None
    last_activity: datetime = field(default_factory=datetime.now)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0


class WorkQueue:
    """Priority-based work queue with load balancing"""
    
    def __init__(self, max_size: int = 10000):
        """Initialize work queue"""
        self.queues = {
            TaskPriority.CRITICAL: deque(),
            TaskPriority.HIGH: deque(),
            TaskPriority.NORMAL: deque(),
            TaskPriority.LOW: deque()
        }
        self.max_size = max_size
        self.size = 0
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
    
    def put(self, item: WorkItem, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Add item to queue"""
        with self.not_full:
            if self.size >= self.max_size:
                if not block:
                    return False
                if not self.not_full.wait(timeout):
                    return False
            
            self.queues[item.priority].append(item)
            self.size += 1
            self.not_empty.notify()
            return True
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[WorkItem]:
        """Get item from queue (priority order)"""
        with self.not_empty:
            while self.size == 0:
                if not block:
                    return None
                if not self.not_empty.wait(timeout):
                    return None
            
            # Get from highest priority non-empty queue
            for priority in TaskPriority:
                if self.queues[priority]:
                    item = self.queues[priority].popleft()
                    self.size -= 1
                    self.not_full.notify()
                    return item
            
            return None
    
    def qsize(self) -> int:
        """Get queue size"""
        with self.lock:
            return self.size
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        with self.lock:
            return self.size == 0


class Worker:
    """Worker for processing tasks"""
    
    def __init__(
        self,
        worker_id: str,
        work_queue: WorkQueue,
        result_queue: asyncio.Queue,
        metrics: WorkerMetrics
    ):
        """Initialize worker"""
        self.worker_id = worker_id
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.metrics = metrics
        self.shutdown_event = threading.Event()
        self.thread = None
        
    def start(self):
        """Start worker thread"""
        self.thread = threading.Thread(target=self._run, name=f"Worker-{self.worker_id}")
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop worker thread"""
        self.shutdown_event.set()
        if self.thread:
            self.thread.join(timeout=5)
    
    def _run(self):
        """Worker main loop"""
        logger.info(f"Worker {self.worker_id} started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get work item
                item = self.work_queue.get(timeout=1)
                
                if item is None:
                    continue
                
                # Update metrics
                self.metrics.state = WorkerState.BUSY
                self.metrics.current_task = item.id
                self.metrics.last_activity = datetime.now()
                item.started_at = datetime.now()
                
                # Process item
                try:
                    # Execute with timeout
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(item.func, *item.args, **item.kwargs)
                        item.result = future.result(timeout=item.timeout)
                    
                    item.completed_at = datetime.now()
                    
                    # Update metrics
                    processing_time = (item.completed_at - item.started_at).total_seconds()
                    self.metrics.tasks_completed += 1
                    self.metrics.total_processing_time += processing_time
                    self.metrics.avg_processing_time = (
                        self.metrics.total_processing_time / self.metrics.tasks_completed
                    )
                    
                except concurrent.futures.TimeoutError:
                    item.error = TimeoutError(f"Task {item.id} timed out after {item.timeout}s")
                    self.metrics.tasks_failed += 1
                    logger.warning(f"Task {item.id} timed out")
                    
                except Exception as e:
                    item.error = e
                    self.metrics.tasks_failed += 1
                    logger.error(f"Task {item.id} failed: {e}")
                    
                    # Retry if applicable
                    if item.retry_count < item.max_retries:
                        item.retry_count += 1
                        self.work_queue.put(item)
                        logger.info(f"Retrying task {item.id} (attempt {item.retry_count})")
                        continue
                
                # Put result in queue
                asyncio.run_coroutine_threadsafe(
                    self.result_queue.put(item),
                    asyncio.get_event_loop()
                )
                
                # Update state
                self.metrics.state = WorkerState.IDLE
                self.metrics.current_task = None
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                self.metrics.state = WorkerState.ERROR
        
        self.metrics.state = WorkerState.SHUTDOWN
        logger.info(f"Worker {self.worker_id} stopped")


class ParallelProcessor:
    """
    Manages parallel processing of radiating system tasks with
    thread pool management and load balancing.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize ParallelProcessor
        
        Args:
            max_workers: Maximum number of workers (default: CPU count)
        """
        # Determine worker count
        if max_workers is None:
            settings = get_radiating_settings()
            max_workers = settings.get('max_parallel_workers', multiprocessing.cpu_count())
        
        self.max_workers = max_workers
        self.redis_client = get_redis_client()
        
        # Work management
        self.work_queue = WorkQueue(max_size=10000)
        self.result_queue: asyncio.Queue = asyncio.Queue()
        self.pending_tasks: Dict[str, WorkItem] = {}
        
        # Worker pool
        self.workers: List[Worker] = []
        self.worker_metrics: Dict[str, WorkerMetrics] = {}
        
        # Load balancing
        self.load_balancer = LoadBalancer(self.worker_metrics)
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_queue_time': 0.0,
            'avg_processing_time': 0.0,
            'peak_queue_size': 0
        }
        
        # Start workers
        self._start_workers()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_workers(self):
        """Start worker threads"""
        for i in range(self.max_workers):
            worker_id = f"worker-{i}"
            metrics = WorkerMetrics(worker_id=worker_id)
            self.worker_metrics[worker_id] = metrics
            
            worker = Worker(
                worker_id=worker_id,
                work_queue=self.work_queue,
                result_queue=self.result_queue,
                metrics=metrics
            )
            
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.max_workers} workers")
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        asyncio.create_task(self._result_processor())
        asyncio.create_task(self._metrics_monitor())
        asyncio.create_task(self._load_balancer_task())
    
    async def process(
        self,
        func: Callable,
        items: List[Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        batch_size: Optional[int] = None,
        timeout: float = 30.0
    ) -> List[Any]:
        """
        Process items in parallel
        
        Args:
            func: Function to apply to each item
            items: Items to process
            priority: Task priority
            batch_size: Batch size for processing
            timeout: Timeout per item
            
        Returns:
            List of results
        """
        if not items:
            return []
        
        # Determine batch size
        if batch_size is None:
            batch_size = max(1, len(items) // (self.max_workers * 2))
        
        # Create work items
        work_items = []
        
        if batch_size > 1:
            # Process in batches
            batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
            
            for i, batch in enumerate(batches):
                item_id = f"batch-{time.time()}-{i}"
                work_item = WorkItem(
                    id=item_id,
                    func=self._process_batch,
                    args=(func, batch),
                    kwargs={},
                    priority=priority,
                    timeout=timeout * len(batch)
                )
                work_items.append(work_item)
                self.pending_tasks[item_id] = work_item
        else:
            # Process individually
            for i, item in enumerate(items):
                item_id = f"item-{time.time()}-{i}"
                work_item = WorkItem(
                    id=item_id,
                    func=func,
                    args=(item,),
                    kwargs={},
                    priority=priority,
                    timeout=timeout
                )
                work_items.append(work_item)
                self.pending_tasks[item_id] = work_item
        
        # Submit work items
        for work_item in work_items:
            self.work_queue.put(work_item)
            self.stats['total_tasks'] += 1
        
        # Update peak queue size
        queue_size = self.work_queue.qsize()
        if queue_size > self.stats['peak_queue_size']:
            self.stats['peak_queue_size'] = queue_size
        
        # Wait for results
        results = await self._wait_for_results(work_items)
        
        return results
    
    def _process_batch(self, func: Callable, batch: List[Any]) -> List[Any]:
        """Process a batch of items"""
        results = []
        for item in batch:
            try:
                result = func(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                results.append(None)
        return results
    
    async def _wait_for_results(self, work_items: List[WorkItem]) -> List[Any]:
        """Wait for work items to complete"""
        pending_ids = {item.id for item in work_items}
        results = {}
        
        while pending_ids:
            # Check for completed items
            completed = []
            
            for item_id in pending_ids:
                if item_id in self.pending_tasks:
                    item = self.pending_tasks[item_id]
                    if item.completed_at is not None or item.error is not None:
                        completed.append(item_id)
                        
                        if item.error:
                            results[item_id] = None
                        else:
                            results[item_id] = item.result
            
            # Remove completed items
            for item_id in completed:
                pending_ids.remove(item_id)
                del self.pending_tasks[item_id]
            
            if pending_ids:
                await asyncio.sleep(0.1)
        
        # Return results in order
        return [results.get(item.id) for item in work_items]
    
    async def _result_processor(self):
        """Process completed work items"""
        while True:
            try:
                # Get completed item
                item = await self.result_queue.get()
                
                # Update statistics
                if item.error:
                    self.stats['failed_tasks'] += 1
                else:
                    self.stats['completed_tasks'] += 1
                    
                    # Update timing statistics
                    if item.started_at and item.created_at:
                        queue_time = (item.started_at - item.created_at).total_seconds()
                        self.stats['avg_queue_time'] = (
                            (self.stats['avg_queue_time'] * (self.stats['completed_tasks'] - 1) + queue_time) /
                            self.stats['completed_tasks']
                        )
                    
                    if item.completed_at and item.started_at:
                        processing_time = (item.completed_at - item.started_at).total_seconds()
                        self.stats['avg_processing_time'] = (
                            (self.stats['avg_processing_time'] * (self.stats['completed_tasks'] - 1) + processing_time) /
                            self.stats['completed_tasks']
                        )
                
            except Exception as e:
                logger.error(f"Error processing result: {e}")
                await asyncio.sleep(1)
    
    async def _metrics_monitor(self):
        """Monitor worker metrics"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Check worker health
                for worker_id, metrics in self.worker_metrics.items():
                    # Check for stuck workers
                    if metrics.state == WorkerState.BUSY:
                        idle_time = (datetime.now() - metrics.last_activity).total_seconds()
                        if idle_time > 60:  # Stuck for more than 1 minute
                            logger.warning(f"Worker {worker_id} may be stuck on task {metrics.current_task}")
                    
                    # Check for error state
                    if metrics.state == WorkerState.ERROR:
                        logger.error(f"Worker {worker_id} in error state")
                
                # Log overall statistics
                active_workers = sum(
                    1 for m in self.worker_metrics.values()
                    if m.state == WorkerState.BUSY
                )
                
                logger.debug(
                    f"Parallel Processor: "
                    f"Active workers: {active_workers}/{self.max_workers}, "
                    f"Queue size: {self.work_queue.qsize()}, "
                    f"Completed: {self.stats['completed_tasks']}, "
                    f"Failed: {self.stats['failed_tasks']}"
                )
                
            except Exception as e:
                logger.error(f"Error in metrics monitor: {e}")
    
    async def _load_balancer_task(self):
        """Load balancing task"""
        while True:
            try:
                await asyncio.sleep(5)  # Balance every 5 seconds
                
                # Get load distribution
                load_distribution = self.load_balancer.get_load_distribution()
                
                # Check for imbalance
                if self.load_balancer.is_imbalanced():
                    logger.info("Load imbalance detected, rebalancing...")
                    # In a real implementation, we might redistribute work here
                
            except Exception as e:
                logger.error(f"Error in load balancer: {e}")
    
    async def map_reduce(
        self,
        map_func: Callable,
        reduce_func: Callable,
        items: List[Any],
        initial_value: Any = None
    ) -> Any:
        """
        Map-reduce pattern for parallel processing
        
        Args:
            map_func: Function to map over items
            reduce_func: Function to reduce results
            items: Items to process
            initial_value: Initial value for reduction
            
        Returns:
            Reduced result
        """
        # Map phase
        mapped_results = await self.process(map_func, items)
        
        # Reduce phase
        if initial_value is not None:
            result = initial_value
        else:
            result = mapped_results[0] if mapped_results else None
            mapped_results = mapped_results[1:]
        
        for item in mapped_results:
            if item is not None:
                result = reduce_func(result, item)
        
        return result
    
    def shutdown(self):
        """Shutdown parallel processor"""
        logger.info("Shutting down parallel processor")
        
        # Stop workers
        for worker in self.workers:
            worker.stop()
        
        logger.info("Parallel processor shutdown complete")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            'workers': {
                worker_id: {
                    'state': metrics.state.value,
                    'tasks_completed': metrics.tasks_completed,
                    'tasks_failed': metrics.tasks_failed,
                    'avg_processing_time': metrics.avg_processing_time,
                    'current_task': metrics.current_task
                }
                for worker_id, metrics in self.worker_metrics.items()
            },
            'queue': {
                'size': self.work_queue.qsize(),
                'peak_size': self.stats['peak_queue_size']
            },
            'tasks': {
                'total': self.stats['total_tasks'],
                'completed': self.stats['completed_tasks'],
                'failed': self.stats['failed_tasks'],
                'pending': len(self.pending_tasks)
            },
            'performance': {
                'avg_queue_time': self.stats['avg_queue_time'],
                'avg_processing_time': self.stats['avg_processing_time']
            }
        }


class LoadBalancer:
    """Load balancer for worker pool"""
    
    def __init__(self, worker_metrics: Dict[str, WorkerMetrics]):
        """Initialize load balancer"""
        self.worker_metrics = worker_metrics
    
    def get_load_distribution(self) -> Dict[str, float]:
        """Get current load distribution"""
        distribution = {}
        
        for worker_id, metrics in self.worker_metrics.items():
            # Calculate load based on state and recent activity
            if metrics.state == WorkerState.BUSY:
                load = 1.0
            elif metrics.state == WorkerState.IDLE:
                # Consider recent activity
                idle_time = (datetime.now() - metrics.last_activity).total_seconds()
                load = max(0, 1 - idle_time / 60)  # Decay over 1 minute
            else:
                load = 0.0
            
            distribution[worker_id] = load
        
        return distribution
    
    def is_imbalanced(self, threshold: float = 0.3) -> bool:
        """Check if load is imbalanced"""
        distribution = self.get_load_distribution()
        
        if not distribution:
            return False
        
        loads = list(distribution.values())
        avg_load = sum(loads) / len(loads)
        
        # Check for significant deviation
        for load in loads:
            if abs(load - avg_load) > threshold:
                return True
        
        return False
    
    def get_least_loaded_worker(self) -> Optional[str]:
        """Get the least loaded worker"""
        distribution = self.get_load_distribution()
        
        if not distribution:
            return None
        
        return min(distribution, key=distribution.get)