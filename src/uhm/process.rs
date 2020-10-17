use super::parser::{AstNode, BlendMode, TargetBuffer, FormulaExpr, Atom, Operator, NormalizeMetric, UnaryOperator, Function};
use std::path::PathBuf;
use rand_pcg::Pcg64;
use rand::prelude::*;

use rustfft::{FFTplanner, num_complex::Complex, num_traits::Zero};

const E: f32 = 2.71828182845904523536;
const PI: f32 = 3.14159265358979323846;

/// .uhm frame type
pub type Frame = [f32; 2048];

/// Converts a list of .uhm commands into sample data
#[derive(Debug, Clone)]
pub struct Processor {
    // We have 3 buffers of the same size, 2 of them are *only* for internal processing, so they do not have external accessors
    // We allow buffer accesses outside of frame_count, but externally only present the frames 0..frame_count of the main buffer
    // We use Vecs instead of fixed-size arrays because otherwise we would overflow the stack *shrug*
    main_buffer: Vec<Frame>,
    aux1_buffer: Vec<Frame>,
    aux2_buffer: Vec<Frame>,
    frame_count: u16, // max: 256 frames

    directory: PathBuf,
    rng: Pcg64,

    randf_buffer: [f32; 256],
    rands_buffer: [f32; 2048],
}

struct FormulaInterpreter<'a> {
    last_result: f32, // Y atom
    processor: &'a mut Processor,
    input: f32, // X|Input atom
    frame: f32,
    table: f32,
    index: f32,
    phase: f32,
}

impl<'a> FormulaInterpreter<'a> {
    fn new(processor: &'a mut Processor) -> FormulaInterpreter<'a> {
        FormulaInterpreter {
            processor,
            last_result: 0.0,
            input: 0.0,
            frame: 0.0,
            table: 0.0,
            index: 0.0,
            phase: 0.0,
        }
    }

    fn interpret(&mut self, formula: &FormulaExpr, prev: &[f32]) -> f32 {
        match formula {
            FormulaExpr::Atom(atom) => self.atom(*atom),
            FormulaExpr::Constant(f) => *f,
            FormulaExpr::Function(func) => self.function(func, prev),
            FormulaExpr::Binary {op, left, right} => {
                let left = self.interpret(left, prev);
                let right = self.interpret(right, prev);
                self.operate(*op, left, right)
            },
            FormulaExpr::Unary {op, item} => {
                let value = self.interpret(item, prev);
                self.unary_operate(*op, value)
            },
            _ => todo!()
        }
    }

    fn atom(&mut self, atom: Atom) -> f32 {
        match atom {
            Atom::Pi => PI,
            Atom::Euler => E,
            Atom::Input => self.input,
            Atom::Phase => self.phase,
            Atom::Table => self.table,
            Atom::Rand => (self.processor.rng.gen::<f32>() % 1.0).abs(),
            Atom::Rands => (self.processor.rands_buffer[self.index as usize] % 1.0).abs(),
            Atom::Randf => (self.processor.randf_buffer[self.frame as usize] % 1.0).abs(),
            _ => todo!("unimplemented atom: {:?}", atom)
        }
    }

    fn operate(&mut self, op: Operator, left: f32, right: f32) -> f32 {
        match op {
            Operator::Minus => left - right,
            Operator::Times => left * right,
            Operator::Power => left.powf(right),
            Operator::Divide => left / right,
            Operator::Modulo => left % right,
            Operator::Plus => left + right,
        }
    }

    fn unary_operate(&mut self, uop: UnaryOperator, value: f32) -> f32 {
        match uop {
            UnaryOperator::Neg => -value,
        }
    }

    fn function(&mut self, func: &Function, prev: &[f32]) -> f32 {
        match func {
            Function::Tanh {value} => self.interpret(value, prev).tanh(),
            Function::Sine {value} => self.interpret(value, prev).sin(),
            Function::Lowpass {value, cutoff, resonance} => {
                let value = self.interpret(value, prev);
                let cutoff = self.interpret(cutoff, prev);
                let resonance = self.interpret(resonance, prev);
                todo!("LPF using an RLC: {} {} {}", value, cutoff, resonance);
            },
            _ => todo!("unimplemented function {:?}", func)
        }
    }
}

impl Processor {
    pub fn new(directory: PathBuf) -> Processor {
        let mut rng = Pcg64::seed_from_u64(0);
        let mut randf_buffer = [0.0; 256];
        let mut rands_buffer = [0.0; 2048];

        Processor::fill_random_buffers(&mut rng, &mut randf_buffer, &mut rands_buffer);

        Processor {
            main_buffer: vec![[0.0; 2048]; 256],
            aux1_buffer: vec![[0.0; 2048]; 256],
            aux2_buffer: vec![[0.0; 2048]; 256],
            frame_count: 256,

            directory,
            // For determinism
            rng,
            randf_buffer,
            rands_buffer,
        }
    }

    fn fill_random_buffers<R: Rng>(rng: &mut R, randfb: &mut[f32; 256], randsb: &mut[f32; 2048]) {
        for float in randfb.iter_mut().chain(randsb.iter_mut()) {
            *float = rng.gen();
        }
    }

    pub fn present(&self) -> Vec<Frame> {
        self.main_buffer[0..self.frame_count as usize].into_iter().cloned().collect()
    }

    fn buffer(&self, tb: TargetBuffer) -> &Vec<Frame> {
        match tb {
            TargetBuffer::Main => &self.main_buffer,
            TargetBuffer::Aux1 => &self.aux1_buffer,
            TargetBuffer::Aux2 => &self.aux2_buffer,
        }
    }

    fn buffer_mut(&mut self, tb: TargetBuffer) -> &mut Vec<Frame> {
        match tb {
            TargetBuffer::Main => &mut self.main_buffer,
            TargetBuffer::Aux1 => &mut self.aux1_buffer,
            TargetBuffer::Aux2 => &mut self.aux2_buffer,
        }
    }

    fn blend(blend_mode: BlendMode, target: &mut [f32], source: &[f32]) {
        for i in 0..2048 {
            match blend_mode {
                BlendMode::Replace => target[i] = source[i],
                BlendMode::Add => target[i] += source[i],
                BlendMode::Sub => target[i] -= source[i],
                BlendMode::Max => target[i] = target[i].max(source[i]),
                BlendMode::Min => target[i] = target[i].min(source[i]),
                _ => todo!()
            }
        }
    }

    // How do you audio normalize???
    fn db_to_lin(f: &[f32], metric: NormalizeMetric, db: f32) -> f32 {
        match metric {
            NormalizeMetric::Rms => {
                // let zdb_factor = Processor::calculate_metric(&f.iter().map(|x| x * target_for_peak).collect::<Vec<_>>(), NormalizeMetric::Rms);
                // println!("{} {}", target_for_peak, zdb_factor);
                10.0f32.powf(db / 10.0) // * zdb_factor
                // A power quanitty
            },
            NormalizeMetric::Peak => 10.0f32.powf(db / 20.0), // An amplitude quantity
            _ => todo!("db_to_lin")
        }
    }

    fn normalize_factor(f: &[f32], metric: NormalizeMetric, db: f32) -> f32 {
        let target = Processor::db_to_lin(f, metric, db);

        match metric {
            NormalizeMetric::Rms => (f.len() as f32 * target.powi(2) / f.iter().map(|x|x.powi(2)).sum::<f32>()).sqrt(),
            NormalizeMetric::Peak => target / f.iter().map(|x| x.abs()).max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap(),
            _ => todo!("normalize_factor")
        }
    }

    fn calculate_metric(f: &[f32], metric: NormalizeMetric) -> f32 {
        match metric {
            NormalizeMetric::Rms => ((1.0 / f.len() as f32) * f.iter().map(|x|x.powi(2)).sum::<f32>()).sqrt(),
            NormalizeMetric::Peak => f.iter().map(|x| x.abs()).max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap(),
            _ => todo!("calculate_metric")
        }
    }

    fn process_fft(&mut self, start: u8, end: u8, blend: BlendMode, target: TargetBuffer, formula: &FormulaExpr, is_forward: bool, lowest: u16, highest: u16, amp_mode: bool) {
        // V. Buggy, we need to find out what exactly Hive is doing...
        let end = std::cmp::min(end, (self.frame_count - 1) as u8);
        let fft = FFTplanner::new(false).plan_fft(2048);
        let ifft = FFTplanner::new(true).plan_fft(2048);
        let target_buffer = self.buffer(target).clone();

        let mut frames = vec![];

        {
            let mut interpreter = FormulaInterpreter::new(self);
            let fmin = std::cmp::min(start, end);
            let fmax = std::cmp::max(start, end);
            for frame in fmin..=fmax {
                let frame = if end < start { end - frame } else { frame };
                let mut target_input: Vec<_> = target_buffer[frame as usize].iter().map(|x| Complex::new(*x, 0.0)).collect();
                let mut target_output: Vec<_> = vec![Complex::zero(); 2048];
                fft.process(&mut target_input, &mut target_output); // Process and get partials for target
                let mut new_amplitudes = [0.0f32; 1025];
                let mut new_phases = [0.0f32; 1025];
                // println!("///\n{:?}\n?\n[{}]", target_buffer[frame as usize], target_output.iter().map(|x| format!("{}", x)).collect::<Vec<_>>().join(","));
                for (i, complex) in target_output.iter().take(1025).enumerate() {
                    new_amplitudes[i] = (complex.re.powi(2) + complex.im.powi(2)).sqrt() * (1.0 / 2048.0);
                    new_phases[i] = complex.im.atan2(complex.re);
                    // println!("Partial {}: {} (theta = {})", i, new_amplitudes[i], new_phases[i]);
                }
                println!("xxx\n{:?}\n><\n{:?}", new_amplitudes, new_phases);
                let min = std::cmp::min(lowest, highest);
                let max = std::cmp::max(lowest, highest);
                for partial in min..=max {
                    let (ref mut target_array, ref mut other_array) = if amp_mode { 
                        (&mut new_amplitudes,  &mut new_phases) 
                    } else { 
                        (&mut new_phases, &mut new_amplitudes)
                    };
                    let partial = if is_forward { 
                        if highest < lowest { max as usize - partial as usize } else { partial as usize } 
                    } else { 
                        if highest < lowest { partial as usize } else { max as usize - partial as usize }
                    };
                    // println!(">> {}", partial);
                    interpreter.input = target_array[partial];
                    interpreter.frame = frame as f32;
                    interpreter.table = (frame - start) as f32 / (end - start) as f32;
                    interpreter.index = partial as f32;
                    interpreter.phase = (partial - min as usize) as f32 / (max - min) as f32;
                    let prev: Vec<f32> = if is_forward { 
                        target_array[0..partial].into_iter().copied().collect() 
                    } else { 
                        target_array[(partial + 1)..].into_iter().copied().rev().collect() 
                    };
                    let result = interpreter.interpret(&formula, prev.as_slice());
                    interpreter.last_result = result;
                    // Because we are editing the spectrum one-sidedly, we have to do this.
                    let result = if amp_mode { result  } else { result };
                    target_array[partial] = result;
                    // Force sine 
                    other_array[partial] = if amp_mode { -std::f32::consts::FRAC_PI_2 } else { 1.0 };
                }
                println!("vvv\n{:?}\nd\n{:?}", new_amplitudes, new_phases);
                let polar:Vec<_> = new_amplitudes.iter().copied().zip(new_phases.iter().copied()).collect();
                let mut new_partials_input: Vec<_> = polar.iter()
                    //.zip(vec![0].into_iter().chain(0..=1023)).map(|(amp, scale)| (1.0 - (scale as f32 / 1024.0)) * amp) // Inverse scale all the amplitudes
                    // We want the real part of the signal to be made up of (positive) sine waves, so that's why
                    // we use amp * sin(phase) for the real part and amp * cos(phase) for the imaginary part
                    // where phase = atan2(real, imag) for the input
                    .map(|(amp, phase)| Complex::new(amp * phase.cos(), amp * phase.sin()))
                    .chain(polar.iter().skip(1).rev().skip(1).map(|(amp, phase)| Complex::new(amp * phase.cos(), -amp * phase.sin())))
                    .collect();
                println!("===\n[{}]", new_partials_input.iter().map(|x| format!("{}", x)).collect::<Vec<_>>().join(","));
                let mut target_output = vec![Complex::zero(); 2048];
                ifft.process(&mut new_partials_input, &mut target_output);
                
                let mut new_frame = [0.0f32; 2048];
                for (i, c) in new_frame.iter_mut().zip(target_output.iter().copied().collect::<Vec<_>>()) {
                    println!("{}", c);
                    *i = c.re;
                }
                // println!("===");
                frames.push(new_frame);
            }
        }

        // Cop out and normalize the peaks of the frame *overall*
        let normalize_factor = frames.iter().map(|f|
            Processor::normalize_factor(f, NormalizeMetric::Peak, 0.0)
        ).max_by(|x, y| x.partial_cmp(y).unwrap());
        
        let norm = if normalize_factor.is_none() && frames.len() > 0 {
            unreachable!("Cannot find normalization factor");
        } else if frames.len() == 0 {
            return
        } else {
            normalize_factor.unwrap()
        };

        for (frame, findex) in frames.into_iter().zip(start..=end) {
            Processor::blend(blend, &mut self.buffer_mut(target)[findex as usize], &frame.iter().map(|x| x * norm).collect::<Vec<_>>());
        }
}

    pub fn interpret(&mut self, node: AstNode) {
        match node {
            AstNode::NumFrames(n) => {
                self.frame_count = n;
            },
            AstNode::Seed(s) => {
                self.rng = Pcg64::seed_from_u64(s);
                Processor::fill_random_buffers(&mut self.rng, &mut self.randf_buffer, &mut self.rands_buffer);
            },
            AstNode::Normalize { start, end, each, target, metric, db } => {
                let end = std::cmp::min(end, (self.frame_count - 1) as u8);
                if each {
                    for frame_id in start..=end {
                        let ref mut frame = self.buffer_mut(target)[frame_id as usize];
                        let factor = Processor::normalize_factor(frame, metric, db);
                        for item in frame.iter_mut() {
                            *item *= factor;
                        }
                    }
                } else {
                    let fid = (start..=end).map(|frame_id| {
                        let ref frame = self.buffer(target)[frame_id as usize];
                        (frame_id, Processor::calculate_metric(frame, metric))
                    }).filter(|(_, x)| x.is_finite()).max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap()).map(|(i, _)| i);

                    if let Some(frame_id) = fid {
                        println!("Using fid {}", frame_id);
                        let frame = self.buffer(target)[frame_id as usize];
                        let factor = Processor::normalize_factor(&frame, metric, db);
                        for frame_id in start..=end {
                            let ref mut frame = &mut self.buffer_mut(target)[frame_id as usize];
                            for item in frame.iter_mut() {
                                *item *= factor;
                            }
                        }
                    } else {
                        // No frame produced a valid value for metric... Should be impossible but...
                        unreachable!("no frame produced a valid metric")
                    }
                }
            },
            AstNode::Wave { start, end, blend, target, formula, is_forward } => {
                let end = std::cmp::min(end, (self.frame_count - 1) as u8);
                let target_buffer = self.buffer(target).clone();
                let mut frames = vec![];
                {
                    let mut interpreter = FormulaInterpreter::new(self);
                    let fmin = std::cmp::min(start, end);
                    let fmax = std::cmp::max(start, end);
                    for frame in fmin..=fmax {
                        let frame = if end < start { end - frame } else { frame };
                        let mut new_frame = [0.0f32; 2048];
                        for sample in 0..2048 {
                            let sample = if is_forward { sample } else { 2047 - sample };
                            interpreter.input = target_buffer[frame as usize][sample];
                            interpreter.frame = frame as f32;
                            interpreter.table = (frame - start) as f32 / (end - start) as f32;
                            interpreter.index = sample as f32;
                            interpreter.phase = sample as f32 / 2047.0;
                            let prev: Vec<f32> = if is_forward { 
                                new_frame[0..sample].into_iter().copied().collect() 
                            } else { 
                                new_frame[(sample + 1)..].into_iter().copied().rev().collect() 
                            };
                            let result = interpreter.interpret(&formula, prev.as_slice());
                            interpreter.last_result = result;
                            new_frame[sample] = result;
                        }
                        frames.push(new_frame);
                    }
                }

                for (frame, findex) in frames.into_iter().zip(start..=end) {
                    Processor::blend(blend, &mut self.buffer_mut(target)[findex as usize], &frame);
                }
            },
            // Support cases where end < start and highest < lowest
            AstNode::Spectrum { start, end, blend, target, formula, is_forward, lowest, highest } => {
                self.process_fft(start, end, blend, target, &formula, is_forward, lowest, highest, true)
            },
            node => todo!("Unhandled AST Node: {:?}", node)
        }
    }
}