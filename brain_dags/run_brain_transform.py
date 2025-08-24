from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python_operator import PythonOperator
from airflow.exceptions import AirflowException, AirflowSkipException
from datetime import datetime, timedelta
from docker.types import Mount
from dotenv import dotenv_values
from airflow.models import DagRun
from airflow.utils.state import State
from airflow.utils.session import create_session

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    # 'retries': 1,
    # 'retry_delay': timedelta(minutes=5),
}

# Add development flag
IS_DEVELOPMENT = False  # Set to True to skip data loading tasks

transform_env = dotenv_values("/opt/airflow/shared_data/env_files/transform.env")
linage_processed_env = dotenv_values("/opt/airflow/shared_data/env_files/linage_processed.env")
linage_viz_env = dotenv_values("/opt/airflow/shared_data/env_files/linage_viz.env")

def _check_run_docker_output(**context):
    logs = context['ti'].xcom_pull(task_ids='run_brain_transform')
    if 'New data was transformed' not in logs:
        raise AirflowSkipException("Skipping containers because logs do not contain 'Processing data'")

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

with DAG(
    'run_brain_transform',
    default_args=default_args,
    description='A DAG to run a Docker container with AWS credentials and check for active DAG runs',
    schedule_interval='0 * * * *',
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    check_dag_active = PythonOperator(
        task_id='check_dag_active',
        python_callable=_skip_if_running,
        provide_context=True,
    )

    run_docker = DockerOperator(
        task_id='run_brain_transform',
        image='registry.code.fbi.h-da.de/mlops-brain/brain_transform/pipeline_api:latest',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',  # Adjust if using a remote Docker host
        network_mode='frontend',
        mount_tmp_dir=False,
        force_pull=True,
        docker_conn_id='registry.code.fbi.h-da.de',  # Matches the connection ID defined earlier
        mounts=[
            Mount(
                source='/services/brain_infrastructure/airflow2/shared_data/.aws',  # Path relative to airflow-worker container
                target='/app/.aws',  # Path inside the Docker container
                type='bind'
            ),
            Mount(
                source='/services/brain_infrastructure/airflow2/shared_data/etl_processed',
                target='/app/processed',
                type='bind'
            ),
            Mount(
                source='/services/brain_infrastructure/airflow2/shared_data/etl_finished',
                target='/app/viz',
                type='bind'
            ),
        ],
        environment=transform_env,
        do_xcom_push=True,
    )

    check_run_docker_output = PythonOperator(
        task_id='check_run_docker_output',
        python_callable=_check_run_docker_output,
        provide_context=True,
    )

    brain_load_processed = DockerOperator(
        task_id='brain_load_processed',
        image='registry.code.fbi.h-da.de/mlops-brain/brain_data/raw_data_tracker:latest',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='frontend',
        mount_tmp_dir=False,
        force_pull=True,
        docker_conn_id='registry.code.fbi.h-da.de',
        mounts=[
            Mount(
                source='/services/brain_infrastructure/airflow2/shared_data/etl_processed',
                target='/app/collect_point',
                type='bind'
            ),
            Mount(
                source='/services/brain_infrastructure/airflow2/shared_data/.aws',
                target='/root/.aws',
                type='bind'
            ),
        ],
        environment=linage_processed_env,
    ) if not IS_DEVELOPMENT else None

    brain_load_viz = DockerOperator(
        task_id='brain_load_viz',
        image='registry.code.fbi.h-da.de/mlops-brain/brain_data/raw_data_tracker:latest',
        api_version='auto',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='frontend',
        mount_tmp_dir=False,
        docker_conn_id='registry.code.fbi.h-da.de',
        mounts=[
            Mount(
                source='/services/brain_infrastructure/airflow2/shared_data/etl_finished',
                target='/app/collect_point',
                type='bind'
            ),
            Mount(
                source='/services/brain_infrastructure/airflow2/shared_data/.aws',
                target='/root/.aws',
                type='bind'
            ),
        ],
        environment=linage_viz_env,
    ) if not IS_DEVELOPMENT else None

    trigger_model_dag = TriggerDagRunOperator(
        task_id='trigger_model_dag',
        trigger_dag_id='model_trigger_dag',  # The DAG ID of the DAG to trigger
        conf={},  # Optional: pass parameters to the triggered DAG
        wait_for_completion=True,  # Optional: wait for the triggered DAG to complete
    )

    # Modified task dependencies
    if IS_DEVELOPMENT:
        check_dag_active >> run_docker >> check_run_docker_output >> trigger_model_dag
    else:
        check_dag_active >> run_docker >> check_run_docker_output >> [brain_load_processed, brain_load_viz] >> trigger_model_dag
