use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Display;
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};
use std::process::Command;

const SBATCH_PREAMBLE: &str = "#!/usr/bin/env bash";

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn slurm_test() {
        let home_dir = civm_rust_utils::home_dir();

        let mut task = SlurmTask::new(&home_dir.join("slurm_test"), "fancy_test_job", 20)
            .no_requeue()
            .partition("reconstruction")
            .output(&home_dir.join("slurm_test").join("slurm_out"))
            .email("wa41@duke.edu", MailType::Fail);

        let mut job_ids = vec![];

        task.submit_later(1);
        job_ids.push(task.job_id.unwrap());
        task.submit_later(2);
        job_ids.push(task.job_id.unwrap());
        task.submit_later(3);
        job_ids.push(task.job_id.unwrap());
        task.submit_later(4);
        job_ids.push(task.job_id.unwrap());
        task.submit_later(5);
        job_ids.push(task.job_id.unwrap());
        task.submit_later(6);
        job_ids.push(task.job_id.unwrap());

        job_ids.push(4287349283);

        println!("job_ids = {:?}", job_ids);
        // println!("sleeping ...");
        // std::thread::sleep(Duration::from_secs(30));

        let s = JobCollection::from_array(job_ids).state();

        println!("states = {:?}", s);
    }

    #[test]
    fn squeue_test() {

        //job_states(ids)
    }
}

pub trait ClusterTask {
    fn configure_slurm_task(&self) -> SlurmTask;

    fn configure_slurm_task_with_opts(&self, options: Vec<SBatchOption>) -> SlurmTask {
        let mut task = self.configure_slurm_task();
        for opt in options {
            task = task.add_opt(opt)
        }
        task
    }

    //fn slurm_job_id(&mut self) -> &mut Option<u64>;

    fn launch_slurm_now(&mut self) -> u64 {
        let mut task = self.configure_slurm_task();
        task.submit()
        //*self.slurm_job_id() = Some(job_id);
        //job_id
    }
    fn launch_slurm_later(&mut self, seconds_later: usize) -> u64 {
        let mut task = self.configure_slurm_task();
        task.submit_later(seconds_later)
        //*self.slurm_job_id() = Some(job_id);
        //job_id
    }

    fn launch_slurm_later_with_opts(
        &mut self,
        seconds_later: usize,
        opts: Vec<SBatchOption>,
    ) -> u64 {
        let mut task = self.configure_slurm_task_with_opts(opts);
        task.submit_later(seconds_later)
        //*self.slurm_job_id() = Some();
    }

    fn launch_slurm_now_with_opts(&mut self, opts: Vec<SBatchOption>) -> u64 {
        let mut task = self.configure_slurm_task_with_opts(opts);
        task.submit()
        //*self.slurm_job_id() = Some();
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum JobState {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
    Unknown,
}

impl Display for JobState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl JobState {
    pub fn from_str(state_str: &str) -> Self {
        match state_str.to_ascii_lowercase().as_str() {
            "pending" => JobState::Pending,
            "cancelled" => JobState::Cancelled,
            "failed" => JobState::Failed,
            "running" => JobState::Running,
            "completed" => JobState::Completed,
            _ => JobState::Unknown,
        }
    }
    pub fn to_string(&self) -> String {
        format!("{:?}", &self)
    }
}

#[derive(Serialize, Deserialize)]
pub struct SBatchOpts {
    // convert to hash map/set (time permitting :D)
    pub reservation: String,
    pub job_name: String,
    pub no_requeue: bool,
    pub memory: Option<String>,
    pub output: String,
    pub partition: String,
    pub start_delay_sec: Option<u32>,
    pub email: Option<String>,
}

#[derive(Clone)]
pub enum MailType {
    None,
    Begin,
    End,
    Fail,
    Requeue,
    All,
}

impl MailType {
    fn format(&self) -> &str {
        match &self {
            MailType::None => "NONE",
            MailType::Begin => "BEGIN",
            MailType::End => "END",
            MailType::Fail => "FAIL",
            MailType::Requeue => "REQUEUE",
            MailType::All => "ALL",
        }
    }
}

#[derive(Clone)]
pub enum SBatchOption {
    /// Name for the job
    JobName(String),
    /// Memory required per node to run
    MemoryMB(usize),
    /// Directory where standard out will be written to
    Output(PathBuf),
    /// Names of the partitions to be considered when scheduling the job
    Partitions(Vec<String>),
    /// Delay for resource allocation after submitting a task. Commonly used when re-scheduling jobs.
    BeginDelaySec(usize),
    /// Names of reservations for job
    Reservations(Vec<String>), // should be a comma seperated list
    /// Eligibility for job requeuing
    NoRequeue,
    /// email address used for reporting job status
    Email(String, MailType),
    /// some dependency on another job
    Dependency(DependencyType),
    /// array job with a range of task ids
    Array{start_id:usize,end_id:usize}
}

#[derive(Clone)]
pub enum DependencyType {
    After { job_id: u64 },
    AfterOk { job_id: u64 },
}

impl SBatchOption {
    fn format(&self) -> String {
        match &self {
            SBatchOption::JobName(name) => format!("#SBATCH --job-name={}", name),
            SBatchOption::MemoryMB(megabytes) => format!("#SBATCH --mem={}", megabytes),
            SBatchOption::Output(directory) => format!(
                "#SBATCH --output={}",
                directory
                    .join("slurm-%j")
                    .with_extension("out")
                    .to_string_lossy()
            ),
            SBatchOption::Partitions(partition_names) => {
                format!("#SBATCH --partition={}", partition_names.join(","))
            }
            SBatchOption::BeginDelaySec(seconds_later) => {
                format!("#SBATCH --begin=now+{}", seconds_later)
            }
            SBatchOption::Reservations(reservation_names) => {
                format!("#SBATCH --reservation={}", reservation_names.join(","))
            }
            SBatchOption::NoRequeue => String::from("#SBATCH --no-requeue"),
            SBatchOption::Email(address, mail_type) => format!(
                "#SBATCH --mail-type={}\n#SBATCH --mail-user={}",
                mail_type.format(),
                address
            ),
            SBatchOption::Dependency(dep_type) => match dep_type {
                DependencyType::After { job_id } => {
                    format!("#SBATCH --dependency=after:{job_id}")
                }
                DependencyType::AfterOk { job_id } => {
                    format!("#SBATCH --dependency=afterok:{job_id}")
                }
            },
            SBatchOption::Array{start_id,end_id} => {
                format!("#SBATCH --array={}-{}\n#SBATCH --ntasks=1",start_id,end_id)
            },
        }
    }
    // unique identifier used in a hashmap to overwrite options
    fn u_id(&self) -> u16 {
        match &self {
            SBatchOption::JobName(_) => 0,
            SBatchOption::MemoryMB(_) => 1,
            SBatchOption::Output(_) => 2,
            SBatchOption::Partitions(_) => 3,
            SBatchOption::BeginDelaySec(_) => 4,
            SBatchOption::Reservations(_) => 5,
            SBatchOption::NoRequeue => 6,
            SBatchOption::Email(_, _) => 7,
            SBatchOption::Dependency(_) => 8,
            SBatchOption::Array { .. } => 9,
        }
    }
}

pub struct SlurmTask {
    script: PathBuf,
    options: HashMap<u16, SBatchOption>,
    commands: Vec<Command>,
    job_id: Option<u64>,
}

impl SlurmTask {
    pub fn new(write_dir: &Path, job_name: &str, memory_megabytes: usize) -> Self {
        use SBatchOption::*;
        let mut options = HashMap::<u16, SBatchOption>::new();
        let jobname = JobName(job_name.to_string());
        let mem = MemoryMB(memory_megabytes);
        let output = Output(write_dir.to_path_buf());

        options.insert(jobname.u_id(), jobname);
        options.insert(mem.u_id(), mem);
        options.insert(output.u_id(), output);

        let script = write_dir.join(job_name).with_extension("bash");

        Self {
            script,
            options,
            commands: vec![Command::new("hostname")],
            job_id: None,
        }
    }

    pub fn job_id(&self) -> Option<u64> {
        self.job_id
    }

    pub fn submit(&mut self) -> u64 {
        let dir = self.script.parent().expect("file should have a parent dir");
        if !dir.exists() {
            if let Err(err) = create_dir_all(dir) {
                panic!("cannot create {:?} with error {:?}", dir, err);
            }
        }
        civm_rust_utils::write_to_file(&self.script, Some("bash"), &self.print());
        let mut cmd = Command::new("sbatch");
        cmd.arg(&self.script);
        let o = cmd.output().expect("failed to run command");
        let resp = String::from_utf8_lossy(&o.stdout).to_string();
        let nums: Vec<u64> = resp
            .split(' ')
            .flat_map(|str| str.replace('\n', "").parse())
            .collect();
        if nums.is_empty() {
            panic!("no job ids found in slurm response")
        }
        if nums.len() != 1 {
            panic!("multiple ids found in slurm response")
        };
        self.job_id = Some(nums[0]);
        nums[0]
    }

    pub fn submit_later(&mut self, seconds_later: usize) -> u64 {
        let opt = SBatchOption::BeginDelaySec(seconds_later);
        self.options.insert(opt.u_id(), opt);
        self.submit()
    }

    /// append a command to the batch script
    pub fn command(self, cmd: Command) -> Self {
        self.add_cmd(cmd)
    }

    /// specify an email address for sbatch task notifications
    pub fn email(self, email_address: &str, mail_type: MailType) -> Self {
        self.add_opt(SBatchOption::Email(email_address.to_string(), mail_type))
    }

    /// specify no re-queuing
    pub fn no_requeue(self) -> Self {
        self.add_opt(SBatchOption::NoRequeue)
    }

    /// add a job dependency for after start
    pub fn job_dependency_after(self, job_id: u64) -> Self {
        self.add_opt(SBatchOption::Dependency(DependencyType::After { job_id }))
    }

    /// add a job dependency for after ok
    pub fn job_dependency_after_ok(self, job_id: u64) -> Self {
        self.add_opt(SBatchOption::Dependency(DependencyType::AfterOk { job_id }))
    }

    /// specify a start delay for the sbatch task in seconds
    pub fn begin_delay_sec(self, delay_sec: usize) -> Self {
        self.add_opt(SBatchOption::BeginDelaySec(delay_sec))
    }

    /// specify output directory for the slurm output file
    pub fn output(self, dir: &Path) -> Self {
        if !dir.exists() {
            if let Err(err) = create_dir_all(dir) {
                panic!(
                    "cannot create slurm output dir {:?} with error {:?}",
                    dir, err
                )
            }
        }
        self.add_opt(SBatchOption::Output(dir.to_owned()))
    }

    pub fn array(self, start_id:usize, end_id:usize) -> Self {
        self.add_opt(
            SBatchOption::Array { start_id, end_id }
        )
    }

    /// specify one of many possible partitions associtaed with the sbatch task. Calling this method again will append to the partition list
    pub fn partition(self, partition_name: &str) -> Self {
        if let Some(part) = self.options.get(&3) {
            if let SBatchOption::Partitions(mut parts) = part.to_owned() {
                parts.push(partition_name.to_string());
                self.add_opt(SBatchOption::Partitions(parts))
            } else {
                panic!("expecting field to be partition!")
            }
        } else {
            self.add_opt(SBatchOption::Partitions(vec![partition_name.to_string()]))
        }
    }

    /// specify one of many possible reservation associtaed with the sbatch task. Calling this method again will append to the reservation list
    pub fn reservation(self, reservation_name: &str) -> Self {
        if let Some(res) = self.options.get(&5) {
            if let SBatchOption::Reservations(mut res) = res.to_owned() {
                res.push(reservation_name.to_string());
                self.add_opt(SBatchOption::Reservations(res))
            } else {
                panic!("expecting field to be a partition!")
            }
        } else {
            self.add_opt(SBatchOption::Reservations(vec![
                reservation_name.to_string()
            ]))
        }
    }

    fn add_opt(mut self, opt: SBatchOption) -> Self {
        self.options.insert(opt.u_id(), opt);
        self
    }

    fn add_cmd(mut self, cmd: Command) -> Self {
        self.commands.push(cmd);
        self
    }

    fn print(&self) -> String {
        let mut lines = vec![];
        lines.push(SBATCH_PREAMBLE.to_string());
        self.options
            .iter()
            .for_each(|(_, opt)| lines.push(opt.format()));
        self.commands
            .iter()
            .for_each(|cmd| lines.push(format!("{:?}", cmd)));
        let mut str = lines.join("\n");
        str.push('\n');
        str
    }
}

pub struct JobCollection {
    job_ids: Vec<u64>,
}

impl Default for JobCollection {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Vec<u64>> for JobCollection {
    fn from(value: Vec<u64>) -> Self {
        Self { job_ids: value }
    }
}

impl JobCollection {
    pub fn new() -> Self {
        Self {
            job_ids: Vec::<u64>::new(),
        }
    }

    pub fn from_iter(i: impl IntoIterator<Item = u64>) -> Self {
        Self {
            job_ids: i.into_iter().collect(),
        }
    }

    pub fn from_array(job_ids: Vec<u64>) -> Self {
        Self {
            job_ids: job_ids.clone(),
        }
    }

    pub fn from_id(job_id: u64) -> Self {
        Self {
            job_ids: vec![job_id],
        }
    }

    pub fn push(&mut self, job_id: u64) {
        self.job_ids.push(job_id);
    }

    pub fn cancel(&self) -> bool {
        let id_strs: Vec<String> = self.job_ids.iter().map(|id| id.to_string()).collect();
        let mut cmd = Command::new("scancel");
        if let Ok(o) = cmd.args(id_strs).output() {
            if !o.status.success() {
                println!("scancel failed");
                false
            } else {
                true
            }
        } else {
            println!("scancel not found!");
            false
        }
    }

    pub fn state(&self) -> HashMap<u64, JobState> {
        let mut h1 = Self::job_state_squeue(&self.job_ids);
        let remaining_job_ids: Vec<u64> = h1
            .iter()
            .filter_map(|(key, val)| {
                if *val == JobState::Unknown {
                    Some(*key)
                } else {
                    None
                }
            })
            .collect();
        if !remaining_job_ids.is_empty() {
            let h2 = Self::job_state_sacct(&remaining_job_ids);
            h2.iter().for_each(|(key, val)| {
                h1.insert(*key, *val);
            })
        }
        h1
    }

    fn job_state_sacct(job_ids: &Vec<u64>) -> HashMap<u64, JobState> {
        let jid_str: Vec<String> = job_ids.iter().map(|j_id| j_id.to_string()).collect();
        let jid_str = jid_str.join(",");
        let mut cmd = Command::new("sacct");
        cmd.args(vec![
            "--parsable2",
            "--noheader",
            "--format=job,state",
            "-j",
            &jid_str,
        ]);
        let o = cmd.output().expect("sacct failed to launch");
        let mut h = HashMap::<u64, JobState>::new();
        match o.status.success() {
            true => {
                let stdout = String::from_utf8(o.stdout).expect("unable to parse stdout");
                stdout.lines().for_each(|line| {
                    let split = line
                        .split_once('|')
                        .expect("delimeter | not found in sacct response");
                    // only parses job ids that are just a number (no extensions)
                    match split.0.parse::<u64>() {
                        Ok(job_id) => {
                            let j_state = JobState::from_str(&split.1.to_ascii_lowercase());
                            h.insert(job_id, j_state);
                        }
                        _ => {}
                    }
                });
            }
            false => {
                return HashMap::<u64, JobState>::from_iter(
                    job_ids.iter().map(|id| (*id, JobState::Unknown)),
                )
            }
        }
        job_ids.iter().for_each(|jid| {
            if h.get(jid).is_none() {
                h.insert(*jid, JobState::Unknown);
            }
        });
        h
    }

    fn job_state_squeue(job_ids: &Vec<u64>) -> HashMap<u64, JobState> {
        let mut cmd = Command::new("squeue");
        cmd.arg("--json");
        let o = cmd.output().expect("failed to run squeue");
        if o.status.success() {
            let s = String::from_utf8_lossy(&o.stdout);
            let v: Value = serde_json::from_str(&s).expect("valid json");
            let jobs_array = v
                .get("jobs")
                .expect("there to be a jobs field")
                .as_array()
                .expect("jobs to be an array");

            // store jobs as a hashmap of json references from original value ^
            let mut jobs_hash = HashMap::<u64, &Value>::with_capacity(jobs_array.len());
            jobs_array.iter().for_each(|val| {
                let job_id = val
                    .get("job_id")
                    .expect("job to have id")
                    .as_u64()
                    .expect("value to be an integer");
                jobs_hash.insert(job_id, val);
            });
            let state_hash = HashMap::<u64, JobState>::from_iter(job_ids.iter().map(|jid| {
                if let Some(val) = jobs_hash.get(jid) {
                    let state_str = val
                        .get("job_state")
                        .expect("to have state")
                        .as_str()
                        .expect("value to be string");
                    (*jid, JobState::from_str(state_str))
                } else {
                    (*jid, JobState::Unknown)
                }
            }));
            state_hash
        } else {
            HashMap::<u64, JobState>::from_iter(job_ids.iter().map(|id| (*id, JobState::Unknown)))
        }
    }
}

/*
simple check to see that slurm is installed on the system
sinfo -V
*/
const SLURM_DISABLED: bool = false;

pub fn is_installed() -> bool {
    if SLURM_DISABLED {
        return false;
    }

    let mut cmd = Command::new("sinfo");
    cmd.arg("-V");

    match cmd.output() {
        Err(_) => {
            //println!("slurm not found on system");
            false
        }
        Ok(_) => true,
    }
}
