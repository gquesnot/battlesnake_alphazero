use std::collections::VecDeque;
use std::fmt::Display;

use rand::Rng;

pub struct BoundedDeque<T> {
    pub deque: VecDeque<T>,
    max_len: usize,
}

impl<T> BoundedDeque<T> {
    pub(crate) fn new(max_len: usize) -> BoundedDeque<T> {
        BoundedDeque {
            deque: VecDeque::new(),
            max_len,
        }
    }

    pub fn append(&mut self, other: Vec<T>) {
        self.deque.extend(other);
        while self.deque.len() > self.max_len {
            self.deque.pop_front();
        }
    }


    pub fn push_back(&mut self, value: T) {
        if self.deque.len() == self.max_len {
            self.deque.pop_front(); // Remove the oldest item if at capacity
        }
        self.deque.push_back(value);
    }

    #[allow(dead_code)]
    fn push_front(&mut self, value: T) {
        if self.deque.len() == self.max_len {
            self.deque.pop_back(); // Remove the newest item if at capacity
        }
        self.deque.push_front(value);
    }

    // Add more methods as needed, like pop_front, pop_back, etc.
}


pub fn choose_index_based_on_probability(probabilities: &[f32]) -> usize {
    let mut rng = rand::thread_rng();
    let mut cumulative_probabilities: Vec<f32> = Vec::new();
    let mut sum = 0.0;

    // Create a cumulative sum array
    for &prob in probabilities {
        sum += prob;
        cumulative_probabilities.push(sum);
    }

    // Generate a random number in the range 0.0 to sum
    let random_num = rng.gen_range(0.0..sum);

    // Find the index where the random number fits in the cumulative array
    cumulative_probabilities.iter().position(|&cum_prob| random_num <= cum_prob).unwrap()
}


pub struct AverageMeter {
    val: f32,
    avg: f32,
    sum: f32,
    count: usize,
}

impl AverageMeter {
    pub fn new() -> Self {
        Self {
            val: 0.0,
            avg: 0.0,
            sum: 0.0,
            count: 0,
        }
    }

    pub fn update(&mut self, val: f32, n: usize) {
        self.val = val;
        self.sum += val * n as f32;
        self.count += n;
        self.avg = self.sum / self.count as f32;
    }
}

impl Default for AverageMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for AverageMeter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.2e}", self.avg)
    }
}