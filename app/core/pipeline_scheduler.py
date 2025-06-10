"""
Pipeline Scheduler Service
Monitors pipeline schedules and executes them when due.
"""
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import text

from app.core.db import SessionLocal
from app.core.pipeline_executor import PipelineExecutor
from app.core.pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)


class PipelineSchedulerService:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.pipeline_executor = PipelineExecutor()
        self.pipeline_manager = PipelineManager()
        self.active_jobs = {}  # Track job IDs by pipeline ID
        
    async def start(self):
        """Start the scheduler service."""
        try:
            self.scheduler.start()
            logger.info("Pipeline scheduler service started")
            
            # Load existing schedules
            await self.load_active_schedules()
            
            # Start monitoring loop
            asyncio.create_task(self.monitor_schedules())
            
        except Exception as e:
            logger.error(f"Failed to start scheduler service: {str(e)}")
    
    async def stop(self):
        """Stop the scheduler service."""
        self.scheduler.shutdown()
        logger.info("Pipeline scheduler service stopped")
    
    async def monitor_schedules(self):
        """Monitor for schedule changes every minute."""
        while True:
            try:
                await self.load_active_schedules()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in schedule monitor: {str(e)}")
                await asyncio.sleep(60)
    
    async def load_active_schedules(self):
        """Load and schedule all active pipeline schedules."""
        db = SessionLocal()
        try:
            # Get all active schedules
            query = text("""
                SELECT ps.*, p.name as pipeline_name
                FROM pipeline_schedules ps
                JOIN agentic_pipelines p ON ps.pipeline_id = p.id
                WHERE ps.is_active = true AND p.is_active = true
            """)
            
            result = db.execute(query)
            schedules = []
            
            for row in result:
                schedule_dict = dict(zip(result.keys(), row))
                schedules.append(schedule_dict)
            
            # Update scheduled jobs
            current_pipeline_ids = set()
            
            for schedule in schedules:
                pipeline_id = schedule['pipeline_id']
                current_pipeline_ids.add(pipeline_id)
                
                # Check if job already exists
                if pipeline_id not in self.active_jobs:
                    self._schedule_pipeline(schedule)
            
            # Remove jobs for deleted/inactive schedules
            for pipeline_id in list(self.active_jobs.keys()):
                if pipeline_id not in current_pipeline_ids:
                    self._unschedule_pipeline(pipeline_id)
                    
        except Exception as e:
            logger.error(f"Failed to load schedules: {str(e)}")
        finally:
            db.close()
    
    def _schedule_pipeline(self, schedule: Dict[str, Any]):
        """Schedule a pipeline based on its configuration."""
        pipeline_id = schedule['pipeline_id']
        pipeline_name = schedule.get('pipeline_name', f'Pipeline {pipeline_id}')
        schedule_type = schedule['schedule_type']
        
        try:
            # Parse schedule config
            import json
            schedule_config = schedule['schedule_config']
            if isinstance(schedule_config, str):
                schedule_config = json.loads(schedule_config)
            
            # Create appropriate trigger
            if schedule_type == 'interval':
                minutes = schedule_config.get('minutes', 60)
                trigger = IntervalTrigger(minutes=minutes)
                
            elif schedule_type == 'cron':
                cron_expr = schedule_config.get('expression', '0 * * * *')
                # Parse cron expression into APScheduler format
                parts = cron_expr.split()
                if len(parts) == 5:
                    trigger = CronTrigger(
                        minute=parts[0],
                        hour=parts[1],
                        day=parts[2],
                        month=parts[3],
                        day_of_week=parts[4]
                    )
                else:
                    logger.error(f"Invalid cron expression for pipeline {pipeline_id}")
                    return
                    
            elif schedule_type == 'one_time':
                run_time = schedule['next_run']
                if isinstance(run_time, str):
                    run_time = datetime.fromisoformat(run_time)
                
                # Only schedule if time is in the future
                if run_time > datetime.now():
                    trigger = DateTrigger(run_date=run_time)
                else:
                    logger.info(f"One-time schedule for pipeline {pipeline_id} is in the past")
                    return
            else:
                logger.error(f"Unknown schedule type: {schedule_type}")
                return
            
            # Add job to scheduler
            job = self.scheduler.add_job(
                self._execute_pipeline,
                trigger,
                args=[pipeline_id, pipeline_name],
                id=f"pipeline_{pipeline_id}",
                name=f"Pipeline: {pipeline_name}",
                replace_existing=True
            )
            
            self.active_jobs[pipeline_id] = job.id
            logger.info(f"Scheduled pipeline {pipeline_id} ({pipeline_name}) with {schedule_type} trigger")
            
        except Exception as e:
            logger.error(f"Failed to schedule pipeline {pipeline_id}: {str(e)}")
    
    def _unschedule_pipeline(self, pipeline_id: int):
        """Remove a pipeline from the scheduler."""
        if pipeline_id in self.active_jobs:
            job_id = self.active_jobs[pipeline_id]
            try:
                self.scheduler.remove_job(job_id)
                del self.active_jobs[pipeline_id]
                logger.info(f"Unscheduled pipeline {pipeline_id}")
            except Exception as e:
                logger.error(f"Failed to unschedule pipeline {pipeline_id}: {str(e)}")
    
    async def _execute_pipeline(self, pipeline_id: int, pipeline_name: str):
        """Execute a scheduled pipeline."""
        logger.info(f"Executing scheduled pipeline {pipeline_id} ({pipeline_name})")
        
        try:
            # Prepare input data for scheduled execution
            input_data = {
                "query": f"Scheduled execution at {datetime.now().isoformat()}",
                "conversation_history": [],
                "scheduled": True
            }
            
            # Execute pipeline
            result = await self.pipeline_executor.execute_pipeline(
                pipeline_id=pipeline_id,
                input_data=input_data,
                trigger_type="scheduled"
            )
            
            logger.info(f"Successfully executed scheduled pipeline {pipeline_id}")
            
            # Update next run time for recurring schedules
            await self._update_next_run(pipeline_id)
            
        except Exception as e:
            logger.error(f"Failed to execute scheduled pipeline {pipeline_id}: {str(e)}")
    
    async def _update_next_run(self, pipeline_id: int):
        """Update next run time for recurring schedules."""
        db = SessionLocal()
        try:
            # Get current schedule
            query = text("""
                SELECT schedule_type, schedule_config 
                FROM pipeline_schedules 
                WHERE pipeline_id = :pipeline_id
            """)
            
            result = db.execute(query, {"pipeline_id": pipeline_id})
            row = result.fetchone()
            
            if row:
                schedule_type = row[0]
                
                # Only update for recurring schedules
                if schedule_type in ['interval', 'cron']:
                    # Calculate next run
                    next_run = self.pipeline_manager._calculate_next_run(
                        schedule_type, 
                        json.loads(row[1]) if isinstance(row[1], str) else row[1]
                    )
                    
                    # Update database
                    update_query = text("""
                        UPDATE pipeline_schedules
                        SET next_run = :next_run
                        WHERE pipeline_id = :pipeline_id
                    """)
                    
                    db.execute(update_query, {
                        "pipeline_id": pipeline_id,
                        "next_run": next_run
                    })
                    db.commit()
                    
        except Exception as e:
            logger.error(f"Failed to update next run for pipeline {pipeline_id}: {str(e)}")
            db.rollback()
        finally:
            db.close()


# Global scheduler instance
pipeline_scheduler = PipelineSchedulerService()