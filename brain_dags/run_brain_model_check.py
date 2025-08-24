from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python_operator import BranchPythonOperator, PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime
import os
from docker.types import Mount
from dotenv import dotenv_values
from airflow.models import DagRun
from airflow.utils.state import State
from airflow.exceptions import AirflowSkipException
from airflow.utils.session import create_session

# Development flags to bypass model trigger check
DEV_SKIP_TRIGGER = False
DEV_SKIP_MODEL = False
DEV_FORCE_TRAINING = False  # Set to True to force model training unconditionally
model_env = dotenv_values("/opt/airflow/shared_data/env_files/model.env")
trigger_env = dotenv_values("/opt/airflow/shared_data/env_files/model_trigger.env")

def check_nfiles_mismatch(**context):
    if DEV_FORCE_TRAINING:
        return 'start_model_training'
    if DEV_SKIP_TRIGGER:
        return 'start_model_training'
    if DEV_SKIP_MODEL:
        return 'end_dag'
    ti = context['ti']
    log_output = ti.xcom_pull(task_ids='run_model_trigger')
    if log_output and any("File counts differ between the dataset commit and master." in line for line in log_output):
        return 'start_model_training'
    else:
        return 'end_dag'

def _skip_if_running(**kwargs):
    dag_id = kwargs['dag_run'].dag_id
    run_id = kwargs['dag_run'].run_id
    with create_session() as session:
        running_count = session.query(DagRun).filter(
            DagRun.dag_id == dag_id,
            DagRun.state == State.RUNNING,
            DagRun.run_id != run_id
        ).count()
    if running_count > 0:
        raise AirflowSkipException("Skipping because another instance of this DAG is already running")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 0,
}

with DAG('model_trigger_dag', default_args=default_args, schedule_interval=None) as dag:
    """
    This DAG triggers a model training process if there is a mismatch in the number of files
    between the dataset commit and the master branch. It first checks if another instance of
    this DAG is already running.
    """

    check_dag_active = PythonOperator(
        task_id='check_dag_active',
        python_callable=_skip_if_running,
        provide_context=True
    )

    run_model_trigger = DockerOperator(
        task_id='run_model_trigger',
        image='registry.code.fbi.h-da.de/mlops-brain/brain_model/trigger:latest',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='frontend',
        environment=trigger_env,
        xcom_all=True,
        docker_conn_id='registry.code.fbi.h-da.de',
        force_pull=True,
        mount_tmp_dir=False,
    )

    check_mismatch = BranchPythonOperator(
        task_id='check_mismatch',
        python_callable=check_nfiles_mismatch,
        provide_context=True
    )

    start_model_training = DockerOperator(
        task_id='start_model_training',
        image='registry.code.fbi.h-da.de/mlops-brain/brain_model/app:latest',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='frontend',
        mount_tmp_dir=False,
        force_pull=True,
        docker_conn_id='registry.code.fbi.h-da.de',
        mounts=[
            Mount(
                source='/services/brain_infrastructure/airflow2/shared_data/.aws',
                target='/root/.aws',
                type='bind'
            )
        ],
        environment=model_env
    )

    end_dag = DummyOperator(
        task_id='end_dag'
    )

    check_dag_active >> run_model_trigger >> check_mismatch >> [start_model_training, end_dag]