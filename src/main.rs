/*
A UHM to WT converter

We process the UHM format and produce the necessary bytes to insert into the WT file

UHM advantages: textual and formulaic
WT advantages: self-contained and can use with Bitwig

Should be called using:
```
$ ./uhmtowt [-i|--i16] [-o|--output <out_filename>] <filename>
```

a.uhm defaults to outputting a.wt
*/
use structopt::StructOpt;

mod uhm;
mod wt;

use std::path::PathBuf;
use std::fs::File;
use anyhow::Result;

#[derive(Debug, StructOpt)]
struct Args {
    #[structopt(long = "i16", short, help = "Outputs i16 .wt file (not present: f32 .wt)")]
    integer_mode: bool,
    #[structopt(short, long, parse(from_os_str))]
    output: Option<PathBuf>,
    #[structopt(parse(from_os_str))]
    filename: PathBuf,
}

#[paw::main]
fn main(args: Args) -> Result<()> {
    if !args.filename.is_file() {
        panic!("Input file does not exist.")
    } else if args.filename.extension().map_or("", |x| x.to_str().unwrap_or("")) != "uhm" {
        panic!("Input file is not a .uhm file")
    }

    eprintln!("Opening {:?}...", args.filename);

    let mut uhm_file = File::open(&args.filename)?;
    let mut uhm_contents = String::new();

    {
        use std::io::Read;
        uhm_file.read_to_string(&mut uhm_contents)?;
        drop(uhm_file);
    }

    eprintln!("Parsing and interpreting {:?}...", args.filename);

    let uhm_lexer = uhm::Lexer::new(uhm_contents.as_str());
    let uhm_parser = uhm::Parser::new(uhm_lexer);
    // All files have a parent directory, so .parent() should always succeed
    let mut processor = uhm::Processor::new(args.filename.parent().unwrap().to_path_buf());

    for ast in uhm_parser {
        let ast = ast?;
        eprintln!("Command: {:?}", ast);
        processor.interpret(ast);
    }

    let frames = processor.present();

    let target = args.output.unwrap_or(args.filename.with_extension("wt"));
    let mut file = File::create(&target)?;

    eprintln!("Writing {:?} [{} frames]...", target, frames.len());

    wt::write_wt(&mut file, &frames, args.integer_mode)?;

    eprintln!("Converted {:?} to {:?}", args.filename, target);

    Ok(())
}
