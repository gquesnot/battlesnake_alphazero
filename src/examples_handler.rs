use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use crate::game::Sample;

pub struct ExamplesHandler {
    root_path: PathBuf,
    max_examples: usize,
    current_index: Option<usize>,
    base_indexes: Vec<usize>,
    loaded_indexes: Vec<usize>,
    pub examples: Vec<Vec<Sample>>,
}


impl ExamplesHandler {
    pub fn new(path: String, max_examples: usize) -> ExamplesHandler {
        let root_path = PathBuf::from(path);
        let mut indexes: Vec<usize> = vec![];
        for entry in fs::read_dir(&root_path).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_file() {
                let file_name = path.file_name().unwrap().to_str().unwrap();
                if !file_name.starts_with("examples_") {
                    continue;
                }
                let index = file_name.split('_').collect::<Vec<&str>>()[1].parse::<usize>().unwrap();
                indexes.push(index);
            }
        }
        indexes.sort();
        println!("Found Examples {:?}", indexes);
        ExamplesHandler {
            root_path,
            max_examples,
            current_index: indexes.last().cloned(),
            base_indexes: indexes,
            examples: vec![],
            loaded_indexes: vec![],
        }
    }


    pub fn load_examples(&mut self) {
        self.examples = vec![];
        self.loaded_indexes = vec![];
        let mut reversed_index = self.base_indexes.clone();
        reversed_index.reverse();
        let mut to_load_indexes= reversed_index.into_iter().take(self.max_examples).collect::<Vec<usize>>();
        to_load_indexes.reverse();
        print!("Loading examples {:?}", to_load_indexes);
        for index in &to_load_indexes {
            self.loaded_indexes.push(*index);
            let file = File::open(&self.root_path.join(format!("examples_{}", index))).unwrap();
            let reader = BufReader::new(file);
            self.examples.push(bincode::deserialize_from(reader).unwrap());
        }
        println!("{}{}", "\rExamples Loaded", " ".repeat(to_load_indexes.len()));
    }


    pub fn save_example(&mut self, example: Vec<Sample>) {
        let new_index = match self.current_index {
            Some(index) => index + 1,
            None => 0,
        };
        print!("Saving example {}...", new_index);
        let file = File::create(self.root_path.join(format!("examples_{}", new_index))).unwrap();
        println!("\rExample {} saved    ", new_index);
        bincode::serialize_into(file, &example).unwrap();
        self.examples.push(example);
        self.current_index = Some(new_index);
        self.base_indexes.push(new_index);
        self.loaded_indexes.push(new_index);
        if self.loaded_indexes.len() > self.max_examples {
            println!("Removing example {}...", self.loaded_indexes[0]);
            self.examples.remove(0);
            fs::remove_file(self.root_path.join(format!("examples_{}", self.loaded_indexes[0]))).unwrap();
            self.loaded_indexes.remove(0);
        }
    }
}