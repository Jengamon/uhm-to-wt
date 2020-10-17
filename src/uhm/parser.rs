//! Parse the token stream, and produce a list of commands

use super::lexer::{Lexer, TokenKind, LexErrorKind};
use thiserror::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::iter::Peekable;

mod formula;

pub use formula::{FormulaExpr, Atom, UnaryOperator, Operator, Function};

#[derive(Debug, Clone, Copy)]
pub enum BlendMode {
    Replace, // source replaces target
    Add, // source added to target
    Sub, // source subtracted from target
    Multiply, // Target is multiplied by source
    MultiplyAbs, // (x + fabs(y))
    Divide, // sign(x * y) * (1 - fabs(x)) * (1 - fabs(y))
    DivideAbs, // sign(x) * (1 - fabs(x)) * (1 - fabs(y))
    Min, // minimum, the smaller absolute value
    Max, // maximum, the larger absolute value
}

#[derive(Debug, Clone, Copy)]
pub enum NormalizeMetric {
    Average,
    Ptp,
    Peak,
    Rms,
}

#[derive(Debug, Clone, Copy)]
pub enum TargetBuffer {
    Main,
    Aux1,
    Aux2,
}

#[derive(Debug, Clone)]
pub enum AstNode {
    NumFrames(u16),
    Seed(u64),
    Wave {
        start: u8,
        end: u8,
        blend: BlendMode,
        is_forward: bool,
        formula: FormulaExpr,
        target: TargetBuffer,
    },
    Spectrum {
        start: u8,
        end: u8,
        lowest: u16,
        highest: u16,
        blend: BlendMode,
        is_forward: bool,
        formula: FormulaExpr,
        target: TargetBuffer,
    },
    Normalize {
        start: u8,
        end: u8,
        metric: NormalizeMetric,
        db: f32,
        each: bool, // true - normalize each frame, false - normalize using max of *all* frames within bounds
        target: TargetBuffer,
    }
}

#[derive(Debug)]
pub enum ParseErrorKind {
    UnexpectedEnd,
    LexerError(LexErrorKind),
    UnexpectedToken(TokenKind),
    NumberOutOfBounds(f32),
    MissingFormula,
}

impl Display for ParseErrorKind {
    fn fmt(&self, fmt: &mut Formatter) -> FmtResult {
        match self {
            ParseErrorKind::UnexpectedEnd => write!(fmt, "unexpected end"),
            ParseErrorKind::LexerError(le) => write!(fmt, "lexer error: {}", le),
            ParseErrorKind::UnexpectedToken(tk) => write!(fmt, "unexpected {:?}", tk),
            ParseErrorKind::NumberOutOfBounds(num) => write!(fmt, "number {} out of bounds", num),
            ParseErrorKind::MissingFormula => write!(fmt, "missing formula"),
        }
    }
}


#[derive(Debug, Error)]
#[error("parse error: {kind} on line {line}")]
pub struct ParseError {
    line: usize,
    kind: ParseErrorKind,
}

pub struct Parser<'a> {
    lexer: Peekable<Lexer<'a>>,
    line: usize,
    frame_count: u16,
}

#[macro_export]
macro_rules! expect {
    ($self:expr => $($($pt:pat)|+ $(if $cond:expr)?),+) => {{
        $(
            if let Some(tok) = $self.advance() {
                let tok = tok?;
                match tok {
                    $($pt)|+ $(if $cond)? => {},
                    t => return Err($self.new_error(ParseErrorKind::UnexpectedToken(t)))
                }
            } else {
                return Err($self.new_error(ParseErrorKind::UnexpectedEnd))
            }
        )+
    }}
}

impl<'a> Parser<'a> {
    pub fn new(lexer: Lexer<'a>) -> Parser<'a> {
        Parser {
            lexer: lexer.peekable(),
            line: 1,
            frame_count: 256,
        }
    }

    fn new_error(&self, kind: ParseErrorKind) -> ParseError {
        ParseError {
            line: self.line,
            kind
        }
    }

    fn advance(&mut self) -> Option<Result<TokenKind, ParseError>> {
        match self.lexer.next() {
            Some(tok) => match tok {
                Ok(token) => {
                    self.line = token.line;
                    Some(Ok(token.kind))
                }
                Err(e) => Some(Err(self.new_error(ParseErrorKind::LexerError(e.kind))))
            }
            None => None
        }
    }

    fn peek(&mut self) -> Option<&TokenKind> {
        self.lexer.peek().and_then(|t| t.as_ref().ok().map(|x| &x.kind))
    }

    fn parse_formula(&mut self) -> Result<FormulaExpr, ParseError> {
        expect!(self => TokenKind::StartFormula);
        let formula = {
            let mut fparser = formula::FormulaParser::new(self);
            fparser.parse()
        }?;
        expect!(self => TokenKind::EndFormula);
        Ok(formula)
    }

    fn parse_target(&mut self) -> Result<TargetBuffer, ParseError> {
        match self.advance().ok_or(self.new_error(ParseErrorKind::UnexpectedEnd))?? {
            TokenKind::Main => Ok(TargetBuffer::Main),
            TokenKind::Aux1 => Ok(TargetBuffer::Aux1),
            TokenKind::Aux2 => Ok(TargetBuffer::Aux2),
            t => Err(self.new_error(ParseErrorKind::UnexpectedToken(t)))
        }
    }

    fn parse_direction(&mut self) -> Result<bool, ParseError> {
        match self.advance().ok_or(self.new_error(ParseErrorKind::UnexpectedEnd))?? {
            TokenKind::Forward => Ok(true),
            TokenKind::Backward => Ok(false),
            t => Err(self.new_error(ParseErrorKind::UnexpectedToken(t)))
        }
    }

    fn parse_blend(&mut self) -> Result<BlendMode, ParseError> {
        match self.advance().ok_or(self.new_error(ParseErrorKind::UnexpectedEnd))?? {
            TokenKind::Replace => Ok(BlendMode::Replace),
            TokenKind::Add => Ok(BlendMode::Add),
            TokenKind::Sub => Ok(BlendMode::Sub),
            TokenKind::Multiply => Ok(BlendMode::Multiply),
            TokenKind::MultiplyAbs => Ok(BlendMode::MultiplyAbs),
            TokenKind::Divide => Ok(BlendMode::Divide),
            TokenKind::DivideAbs => Ok(BlendMode::DivideAbs),
            TokenKind::Min => Ok(BlendMode::Min),
            TokenKind::Max => Ok(BlendMode::Max),
            t => Err(self.new_error(ParseErrorKind::UnexpectedToken(t)))
        }
    }

    fn parse_base(&mut self) -> Result<bool, ParseError> {
        match self.advance().ok_or(self.new_error(ParseErrorKind::UnexpectedEnd))?? {
            TokenKind::Each => Ok(true),
            TokenKind::All => Ok(false),
            t => Err(self.new_error(ParseErrorKind::UnexpectedToken(t)))
        }
    }

    fn parse_metric(&mut self) -> Result<NormalizeMetric, ParseError> {
        match self.advance().ok_or(self.new_error(ParseErrorKind::UnexpectedEnd))?? {
            TokenKind::Peak => Ok(NormalizeMetric::Peak),
            TokenKind::Average => Ok(NormalizeMetric::Average),
            TokenKind::Ptp => Ok(NormalizeMetric::Ptp),
            TokenKind::Rms => Ok(NormalizeMetric::Rms),
            t => Err(self.new_error(ParseErrorKind::UnexpectedToken(t)))
        }
    }

    fn parse_number(&mut self) -> Result<f32, ParseError> {
        match self.advance().ok_or(self.new_error(ParseErrorKind::UnexpectedEnd))?? {
            TokenKind::Float(f) => Ok(f),
            TokenKind::Integer(i) => Ok(i as f32),
            t => Err(self.new_error(ParseErrorKind::UnexpectedToken(t)))
        }
    }

    /// Parses a number from [min, max]
    fn parse_bounded_number(&mut self, min: i64, max: i64) -> Result<i64, ParseError> {
        match self.advance().ok_or(self.new_error(ParseErrorKind::UnexpectedEnd))?? {
            TokenKind::Integer(n) => if n >= min && n <= max {
                Ok(n)
            } else {
                Err(self.new_error(ParseErrorKind::NumberOutOfBounds(n as f32)))
            },
            t => Err(self.new_error(ParseErrorKind::UnexpectedToken(t)))
        }
    }

    fn num_frames_command(&mut self) -> Result<AstNode, ParseError> {
        expect!(self => TokenKind::Equal);
        let num_frames = match self.advance().ok_or(self.new_error(ParseErrorKind::UnexpectedEnd))?? {
            TokenKind::Integer(n) => if 0 <= n && n <= 256 {
                n as u16
            } else {
                return Err(self.new_error(ParseErrorKind::NumberOutOfBounds(n as f32)))
            },
            t => return Err(self.new_error(ParseErrorKind::UnexpectedToken(t)))
        };
        self.frame_count = num_frames;
        Ok(AstNode::NumFrames(num_frames))
    }

    fn seed_command(&mut self) -> Result<AstNode, ParseError> {
        expect!(self => TokenKind::Equal);
        let seed = match self.advance().ok_or(self.new_error(ParseErrorKind::UnexpectedEnd))?? {
            TokenKind::Integer(n) => n as u64,
            t => return Err(self.new_error(ParseErrorKind::UnexpectedToken(t)))
        };
        Ok(AstNode::Seed(seed))
    }

    fn wave_command(&mut self) -> Result<AstNode, ParseError> {
        let mut start = 0;
        let mut end = (self.frame_count - 1) as u8;
        let mut blend = BlendMode::Replace;
        let mut is_forward = true;
        let mut formula = None;
        let mut target = TargetBuffer::Main;

        // Do the parsing dance
        loop {
            match self.peek() {
                Some(TokenKind::StartFormula) => {
                    formula = Some(self.parse_formula()?)
                },
                Some(TokenKind::Start) => {
                    expect!(self => TokenKind::Start, TokenKind::Equal);
                    start = self.parse_bounded_number(0, u8::MAX as i64)? as u8;
                },
                Some(TokenKind::End) => {
                    expect!(self => TokenKind::End, TokenKind::Equal);
                    end = self.parse_bounded_number(0, u8::MAX as i64)? as u8;
                },
                Some(TokenKind::Target) => {
                    expect!(self => TokenKind::Target, TokenKind::Equal);
                    target = self.parse_target()?;
                },
                Some(TokenKind::Direction) => {
                    expect!(self => TokenKind::Direction, TokenKind::Equal);
                    is_forward = self.parse_direction()?;
                },
                Some(TokenKind::Blend) => {
                    expect!(self => TokenKind::Blend, TokenKind::Equal);
                    blend = self.parse_blend()?;
                },
                _ => break
            }
        }

        match formula {
            Some(formula) => Ok(AstNode::Wave {start, end, blend, is_forward, formula, target}),
            None => Err(self.new_error(ParseErrorKind::MissingFormula))
        }
    }

    fn spectrum_command(&mut self) -> Result<AstNode, ParseError> {
        let mut start = 0;
        let mut end = (self.frame_count - 1) as u8;
        let mut blend = BlendMode::Replace;
        let mut lowest = 1;
        let mut highest = 1023;
        let mut is_forward = true;
        let mut formula = None;
        let mut target = TargetBuffer::Main;

        // Do the parsing dance
        loop {
            match self.peek() {
                Some(TokenKind::StartFormula) => {
                    formula = Some(self.parse_formula()?)
                },
                Some(TokenKind::Start) => {
                    expect!(self => TokenKind::Start, TokenKind::Equal);
                    start = self.parse_bounded_number(0, u8::MAX as i64)? as u8;
                },
                Some(TokenKind::End) => {
                    expect!(self => TokenKind::End, TokenKind::Equal);
                    end = self.parse_bounded_number(0, u8::MAX as i64)? as u8;
                },
                Some(TokenKind::Lowest) => {
                    expect!(self => TokenKind::Lowest, TokenKind::Equal);
                    lowest = self.parse_bounded_number(0, 1023)? as u16;
                },
                Some(TokenKind::Highest) => {
                    expect!(self => TokenKind::Highest, TokenKind::Equal);
                    highest = self.parse_bounded_number(0, 1023)? as u16;
                }
                Some(TokenKind::Target) => {
                    expect!(self => TokenKind::Target, TokenKind::Equal);
                    target = self.parse_target()?;
                },
                Some(TokenKind::Direction) => {
                    expect!(self => TokenKind::Direction, TokenKind::Equal);
                    is_forward = self.parse_direction()?;
                },
                Some(TokenKind::Blend) => {
                    expect!(self => TokenKind::Blend, TokenKind::Equal);
                    blend = self.parse_blend()?;
                },
                _ => break
            }
        }

        match formula {
            Some(formula) => Ok(AstNode::Spectrum {start, end, blend, is_forward, formula, lowest, highest, target}),
            None => Err(self.new_error(ParseErrorKind::MissingFormula))
        }
    }

    fn normalize_command(&mut self) -> Result<AstNode, ParseError> {
        let mut start = 0;
        let mut end = (self.frame_count - 1) as u8;
        let mut target = TargetBuffer::Main;
        let mut db = 0.0;
        let mut metric = NormalizeMetric::Rms;
        let mut each = true;

        loop {
            match self.peek() {
                Some(TokenKind::Start) => {
                    expect!(self => TokenKind::Start, TokenKind::Equal);
                    start = self.parse_bounded_number(0, u8::MAX as i64)? as u8;
                },
                Some(TokenKind::End) => {
                    expect!(self => TokenKind::End, TokenKind::Equal);
                    end = self.parse_bounded_number(0, u8::MAX as i64)? as u8;
                },
                Some(TokenKind::Db) => {
                    expect!(self => TokenKind::Db, TokenKind::Equal);
                    db = self.parse_number()?;
                },
                Some(TokenKind::Target) => {
                    expect!(self => TokenKind::Target, TokenKind::Equal);
                    target = self.parse_target()?;
                },
                Some(TokenKind::Metric) => {
                    expect!(self => TokenKind::Metric, TokenKind::Equal);
                    metric = self.parse_metric()?;
                },
                Some(TokenKind::Base) => {
                    expect!(self => TokenKind::Base, TokenKind::Equal);
                    each = self.parse_base()?;
                },
                _ => break
            }
        }

        Ok(AstNode::Normalize {
            start,
            end,
            target,
            db,
            metric,
            each
        })
    }
}

impl<'a> Iterator for Parser<'a> {
    type Item = Result<AstNode, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.advance()? {
            Ok(token) => match token {
                TokenKind::NumFrames => Some(self.num_frames_command()),
                TokenKind::Seed => Some(self.seed_command()),
                TokenKind::Wave => Some(self.wave_command()),
                TokenKind::Spectrum => Some(self.spectrum_command()),
                TokenKind::Normalize => Some(self.normalize_command()),
                t => Some(Err(self.new_error(ParseErrorKind::UnexpectedToken(t)))),
            },
            Err(pe) => Some(Err(pe))
        }
    }
}