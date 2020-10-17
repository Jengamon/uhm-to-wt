//! Stuff for parsing .uhm files

use std::iter::Iterator;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::str::Chars;
use std::iter::Peekable;
use thiserror::Error;
use lazy_static::lazy_static;
use std::collections::HashMap;


lazy_static! {
    static ref KEYWORDS_MAP: HashMap<String, TokenKind> = {
        let mut map = HashMap::new();

        map.insert("Info".into(), TokenKind::Info);
        map.insert("NumFrames".into(), TokenKind::NumFrames);
        map.insert("Seed".into(), TokenKind::Seed);
        map.insert("Wave".into(), TokenKind::Wave);
        map.insert("Normalize".into(), TokenKind::Normalize);
        map.insert("Spectrum".into(), TokenKind::Spectrum);

        map.insert("Start".into(), TokenKind::Start);
        map.insert("End".into(), TokenKind::End);
        map.insert("Blend".into(), TokenKind::Blend);
        map.insert("Target".into(), TokenKind::Target);
        map.insert("Direction".into(), TokenKind::Direction);
        map.insert("Metric".into(), TokenKind::Metric);
        map.insert("DB".into(), TokenKind::Db);
        map.insert("Base".into(), TokenKind::Base);
        map.insert("Lowest".into(), TokenKind::Lowest);
        map.insert("Highest".into(), TokenKind::Highest);
        map.insert("start".into(), TokenKind::Start);
        map.insert("end".into(), TokenKind::End);
        map.insert("blend".into(), TokenKind::Blend);
        map.insert("target".into(), TokenKind::Target);
        map.insert("direction".into(), TokenKind::Direction);
        map.insert("metric".into(), TokenKind::Metric);
        map.insert("dB".into(), TokenKind::Db);
        map.insert("base".into(), TokenKind::Base);
        map.insert("lowest".into(), TokenKind::Lowest);
        map.insert("highest".into(), TokenKind::Highest);

        map.insert("forward".into(), TokenKind::Forward);
        map.insert("fwd".into(), TokenKind::Forward);
        map.insert("backward".into(), TokenKind::Backward);
        map.insert("bwd".into(), TokenKind::Backward);
        map.insert("main".into(), TokenKind::Main);
        map.insert("aux1".into(), TokenKind::Aux1);
        map.insert("aux2".into(), TokenKind::Aux2);
        map.insert("replace".into(), TokenKind::Replace);
        map.insert("add".into(), TokenKind::Add);
        map.insert("sub".into(), TokenKind::Sub);
        map.insert("multiply".into(), TokenKind::Multiply);
        map.insert("multiplyAbs".into(), TokenKind::MultiplyAbs);
        map.insert("divide".into(), TokenKind::Divide);
        map.insert("divideAbs".into(), TokenKind::DivideAbs);
        map.insert("min".into(), TokenKind::Min);
        map.insert("max".into(), TokenKind::Max);
        map.insert("Main".into(), TokenKind::Main);
        map.insert("Aux1".into(), TokenKind::Aux1);
        map.insert("Aux2".into(), TokenKind::Aux2);
        map.insert("Replace".into(), TokenKind::Replace);
        map.insert("Add".into(), TokenKind::Add);
        map.insert("Sub".into(), TokenKind::Sub);
        map.insert("Multiply".into(), TokenKind::Multiply);
        map.insert("MultiplyAbs".into(), TokenKind::MultiplyAbs);
        map.insert("Divide".into(), TokenKind::Divide);
        map.insert("DivideAbs".into(), TokenKind::DivideAbs);
        map.insert("Min".into(), TokenKind::Min);
        map.insert("Max".into(), TokenKind::Max);
        map.insert("rms".into(), TokenKind::Rms);
        map.insert("RMS".into(), TokenKind::Rms);
        map.insert("Peak".into(), TokenKind::Peak);
        map.insert("peak".into(), TokenKind::Peak);
        map.insert("average".into(), TokenKind::Average);
        map.insert("Average".into(), TokenKind::Average);
        map.insert("Ptp".into(), TokenKind::Ptp);
        map.insert("ptp".into(), TokenKind::Ptp);
        map.insert("All".into(), TokenKind::All);
        map.insert("all".into(), TokenKind::All);
        map.insert("Each".into(), TokenKind::Each);
        map.insert("each".into(), TokenKind::Each);

        map
    };

    static ref FORMULA_KEYWORDS_MAP: HashMap<String, TokenKind> = {
        let mut map = HashMap::new();

        map.insert("sin".into(), TokenKind::Sine);
        map.insert("rand".into(), TokenKind::Rand);
        map.insert("x".into(), TokenKind::Input);
        map.insert("input".into(), TokenKind::Input);
        map.insert("lowpass".into(), TokenKind::Lowpass);
        map.insert("tanh".into(), TokenKind::Tanh);
        map.insert("phase".into(), TokenKind::Phase);
        map.insert("table".into(), TokenKind::Table);
        map.insert("pi".into(), TokenKind::Pi);
        map.insert("e".into(), TokenKind::Euler);

        map
    };
}

#[derive(Debug, Clone)]
pub enum TokenKind {
    Integer(i64),
    Float(f32),
    String(String),
    StartFormula,
    EndFormula,

    // Keywords
    Info,
    NumFrames,
    Seed,
    Wave,
    Normalize,
    Spectrum,

    Start,
    End,
    Blend,
    Direction,
    Target,
    Metric,
    Db,
    Base,
    Lowest,
    Highest,

    Forward,
    Backward,

    Main,
    Aux1,
    Aux2,

    Replace,
    Add,
    Sub,
    Multiply,
    MultiplyAbs,
    Divide,
    DivideAbs,
    Min,
    Max,

    Rms,
    Peak,
    Average,
    Ptp,

    All,
    Each,

    // Formula Keywords
    Sine,
    Rand,
    Phase,
    Table,
    Pi,
    Euler,
    Input,
    Lowpass,
    Tanh,

    // Symbols
    Equal,
    EqualEqual,
    Minus,
    Asterisk,
    ForwardSlash,
    Plus,
    Caret,
    Percent,
    Comma,
    LParen,
    RParen,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub line: usize,
    pub kind: TokenKind,
}

#[derive(Debug)]
pub enum LexErrorKind {
    UnexpectedEOF,
    UnexpectedChar(char),
    UnexpectedString(String),
    ParseFloatError(std::num::ParseFloatError),
    ParseIntError(std::num::ParseIntError),
}

impl Display for LexErrorKind {
    fn fmt(&self, fmt: &mut Formatter) -> FmtResult {
        match self {
            LexErrorKind::UnexpectedEOF => write!(fmt, "unexpected EOF"),
            LexErrorKind::UnexpectedChar(c) => write!(fmt, "unexpected char: {}", c),
            LexErrorKind::UnexpectedString(s) => write!(fmt, "unexpected string: {}", s),
            LexErrorKind::ParseFloatError(pfe) => write!(fmt, "float parse error: {}", pfe),
            LexErrorKind::ParseIntError(pie) => write!(fmt, "integer parse error: {}", pie),
        }
    }
}

#[derive(Debug, Error)]
#[error("lex error: {kind} at line {line}")]
pub struct LexError {
    pub kind: LexErrorKind,
    pub line: usize
}

pub struct Lexer<'a> {
    iter: Peekable<Chars<'a>>,
    line: usize,
    formula_mode: bool,
    string_flag: bool,
}

macro_rules! consume {
    ($self:expr => $($pt:pat)|+ $(if $cond:expr)?) => {
        loop {
            if let Some(c) = $self.peek() {
                match c {
                    $($pt)|+ $(if $cond)? => {drop($self.advance());},
                    _ => break
                }
            } else {
                break
            }
        }
    };

    (expect $self:expr => $($($pt:pat)|+ $(if $cond:expr)?),+) => {{
        $(
            match $self.peek().cloned() {
                $($pt)|+ $(if $cond)? => {drop($self.advance());},
                Some(c) => return Err($self.new_error(LexErrorKind::UnexpectedChar(c))),
                None => return Err($self.new_error(LexErrorKind::UnexpectedEOF))
            }
        )+
    }};

    (count $self:expr => $num:expr) => {
        for _ in 0..$num {
            drop($self.advance());
        }
    }
}

impl<'a> Lexer<'a> {
    pub fn new(s: &'a str) -> Lexer<'a> {
        Lexer {
            line: 1,
            iter: s.chars().peekable(),
            formula_mode: false,
            string_flag: false,
        }
    }

    fn new_error(&self, kind: LexErrorKind) -> LexError {
        LexError {
            kind,
            line: self.line
        }
    }

    fn new_token(&self, kind: TokenKind) -> Token {
        Token {
            kind,
            line: self.line
        }
    }

    fn advance(&mut self) -> Result<char, LexError> {
        match self.iter.next() {
            Some(c) => {
                if c == '\n' { self.line += 1 }
                Ok(c)
            }
            None => Err(self.new_error(LexErrorKind::UnexpectedEOF))
        }
    }

    fn peek(&mut self) -> Option<&char> {
        self.iter.peek()
    }

    fn check_next<F>(&mut self, f: F) -> bool where F: FnOnce(char) -> bool {
        if let Some(c) = self.peek() {
            f(*c)
        } else {
            false
        }
    }

    fn skip_comment(&mut self) -> Result<bool, LexError> {
        if self.formula_mode { // Don't allow comments inside formulas
            return Ok(false)
        }

        let det_char = self.peek();
        if let Some(chr) = det_char {
            match chr {
                '*' => {
                    consume!(count self => 1);
                    // Multiline comment parsing
                    let mut level = 1;
                    while level > 0 {
                        let c = self.advance()?;
                        match c {
                            '*' => if self.check_next(|c| c == '/') {
                                consume!(count self => 1);
                                level -= 1;
                            },
                            '/' => if self.check_next(|c| c == '*') {
                                consume!(count self => 1);
                                level += 1;
                            },
                            _ => { consume!(count self => 1); }
                        }
                    }
                    Ok(true)
                },
                '/' => {
                    // Single line comment parsing
                    consume!(self => c if *c != '\n');
                    Ok(true)
                },
                _ => Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    fn parse_keyword(&mut self, init_c: char) -> Result<Option<Token>, LexError> {
        let mut string = String::new();
        string.push(init_c);

        loop {
            match self.peek() {
                Some('A'..='Z') | Some('a'..='z') | Some('0'..='9') | Some('_') => string.push(self.advance()?),
                _ => break
            }
        }

        let token_type = if !self.formula_mode {
            KEYWORDS_MAP.get(&string).ok_or(self.new_error(LexErrorKind::UnexpectedString(string)))
        } else {
            FORMULA_KEYWORDS_MAP.get(&string).ok_or(self.new_error(LexErrorKind::UnexpectedString(string)))
        }?;

        match token_type {
            // Treat "Info" commands like one line comments
            TokenKind::Info => {
                consume!(self => c if *c != '\n');
                Ok(None)
            },
            kind => Ok(Some(self.new_token(kind.clone())))
        }
    }

    fn parse_number(&mut self, init_c: char, positive: bool) -> Result<Token, LexError> {
        let mut number = String::new();
        number.push(init_c);

        let mut dot = false;
        while let Some(c) = self.peek() {
            match c {
                '0'..='9' => {number.push(self.advance()?);},
                '.' if !dot => {
                    dot = true;
                    number.push(self.advance()?);
                },
                _ => break
            }
        }

        if dot {
            let mult = if positive { 1.0 } else { -1.0 };
            match number.parse::<f32>() {
                Ok(f) => Ok(self.new_token(TokenKind::Float(mult * f))),
                Err(e) => Err(self.new_error(LexErrorKind::ParseFloatError(e)))
            }
        } else {
            let mult = if positive { 1 } else { -1 };
            match number.parse::<i64>() {
                Ok(i) => Ok(self.new_token(TokenKind::Integer(mult * i))),
                Err(e) => Err(self.new_error(LexErrorKind::ParseIntError(e)))
            }
        }
        
    }

    fn parse_string(&mut self) -> Result<Token, LexError> {
        let mut string = String::new();
        while !matches!(self.peek(), Some('"') | Some('\n') | None) {
            string.push(self.advance()?);
        }
        consume!(expect self => Some('"'));
        Ok(self.new_token(TokenKind::String(string)))
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<Token, LexError>;

    fn next(&mut self) -> Option<Self::Item> {
        // If we generate an error on advance here, we just want to return none, bc that means we are at the end of the file
        loop {
            match self.advance().ok()? {
                '/' => match self.skip_comment() {
                    Ok(true) => {},
                    Ok(false) => return Some(Ok(self.new_token(TokenKind::ForwardSlash))),
                    Err(e) => return Some(Err(e)),
                },
                c @ 'A'..='Z' | c @ 'a'..='z' => match self.parse_keyword(c) {
                    Ok(res) => if let Some(tok) = res { return Some(Ok(tok)) },
                    Err(e) => return Some(Err(e))
                },
                c @ '0'..='9' => return Some(self.parse_number(c, true)),
                '=' => if self.check_next(|c| c == '=') {
                    consume!(count self => 1);
                    return Some(Ok(self.new_token(TokenKind::EqualEqual)))
                } else {
                    return Some(Ok(self.new_token(TokenKind::Equal)))
                },
                '"' if self.string_flag => {
                    self.string_flag = false;
                    return Some(self.parse_string())
                },
                '"' if !self.formula_mode => {
                    self.formula_mode = true;
                    return Some(Ok(self.new_token(TokenKind::StartFormula)))
                },
                '"' if self.formula_mode => {
                    self.formula_mode = false;
                    return Some(Ok(self.new_token(TokenKind::EndFormula)))
                },
                '-' if !self.formula_mode => if self.check_next(|c| c.is_digit(10)) {
                    // Allow for direct constant numbers
                    let nc = self.advance().unwrap();
                    return Some(self.parse_number(nc, false))
                } else {
                    return Some(Err(self.new_error(LexErrorKind::UnexpectedChar('-'))))
                },
                '*' => return Some(Ok(self.new_token(TokenKind::Asterisk))),
                ',' => return Some(Ok(self.new_token(TokenKind::Comma))),
                '(' => return Some(Ok(self.new_token(TokenKind::LParen))),
                ')' => return Some(Ok(self.new_token(TokenKind::RParen))),
                '+' => return Some(Ok(self.new_token(TokenKind::Plus))),
                '-' if self.formula_mode => return Some(Ok(self.new_token(TokenKind::Minus))),
                c if c.is_whitespace() => {},
                c => return Some(Err(self.new_error(LexErrorKind::UnexpectedChar(c))))
            }
        }
    }
}