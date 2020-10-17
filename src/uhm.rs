/*
.uhm files consist of 1 - 256 frames of a fixed size 2048

They use lists of commands to specify how to process these in order.

We use 32-bit processing, but may switch to 64-bit later on.
*/

mod lexer;
mod parser;
mod process;

pub use lexer::{LexError, LexErrorKind, Lexer};
pub use process::{Frame, Processor};
pub use parser::{Parser, ParseError, ParseErrorKind, AstNode};