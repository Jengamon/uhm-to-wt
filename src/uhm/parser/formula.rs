//! Pratt parsers for formulas

use super::{LexErrorKind, Lexer, ParseError, ParseErrorKind, TokenKind, Parser};
use crate::expect;

#[derive(Debug, Clone, Copy)]
pub enum Operator {
    Plus,
    Minus,
    Times,
    Divide,
    Power,
    Modulo,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOperator {
    Neg,
}

#[derive(Debug, Clone, Copy)]
pub enum Atom {
    Rand,
    Phase,
    Table,
    Pi,
    Euler,
    Input,
    Rands,
    Randf,
}

#[derive(Debug, Clone)]
pub enum Function {
    Lowpass {
        value: Box<FormulaExpr>,
        cutoff: Box<FormulaExpr>,
        resonance: Box<FormulaExpr>,
    },
    Tanh {
        value: Box<FormulaExpr>,
    },
    Sine {
        value: Box<FormulaExpr>,
    }
}

#[derive(Debug, Clone)]
pub enum FormulaExpr {
    Constant(f32),
    Atom(Atom),
    Unary {
        op: UnaryOperator,
        item: Box<FormulaExpr>
    },
    Binary {
        left: Box<FormulaExpr>,
        right: Box<FormulaExpr>,
        op: Operator
    },
    Function(Function),
}

pub(in super) struct FormulaParser<'a, 'b: 'a> {
    parser: &'a mut Parser<'b>,
}

macro_rules! operator {
    (binary $self:expr, $tok:expr, $left:expr => $(
        $tkp:pat => $op:expr
    ),*$(,)?) => { match $tok {
        $(
            tok @ $tkp => {
                let query = FormulaParser::infix_query(&tok).expect(&format!("failed to implement infix query for {:?}", tok));
                let prec = if query.right { query.precedence - 1 } else { query.precedence };
                let right = Box::new($self.parse_expression(prec)?);
                Ok(FormulaExpr::Binary {
                    op: $op,
                    left: Box::new($left),
                    right
                })
            }
        ),*
        t => Err($self.new_error(t))
    }}
}

#[derive(Debug, Clone, Copy)]
struct Query {
    precedence: u8,
    right: bool
}

impl Query {
    fn new(precedence: u8, right: bool) -> Query {
        Query {precedence, right}
    }
}

impl<'a, 'b> FormulaParser<'a, 'b> {
    pub fn new(parser: &'a mut Parser<'b>) -> FormulaParser<'a, 'b> {
        FormulaParser {
            parser
        }
    }

    fn new_error(&self, token: TokenKind) -> ParseError {
        self.parser.new_error(ParseErrorKind::UnexpectedToken(token))
    }

    pub fn prefix(&mut self, tok: TokenKind) -> Result<FormulaExpr, ParseError> {
        match tok {
            TokenKind::Float(f) => Ok(FormulaExpr::Constant(f)),
            TokenKind::Integer(i) => Ok(FormulaExpr::Constant(i as f32)),
            TokenKind::Rand => Ok(FormulaExpr::Atom(Atom::Rand)),
            TokenKind::Phase => Ok(FormulaExpr::Atom(Atom::Phase)),
            TokenKind::Table => Ok(FormulaExpr::Atom(Atom::Table)),
            TokenKind::Pi => Ok(FormulaExpr::Atom(Atom::Pi)),
            TokenKind::Euler => Ok(FormulaExpr::Atom(Atom::Euler)),
            TokenKind::Input => Ok(FormulaExpr::Atom(Atom::Input)),
            TokenKind::LParen => {
                let expr = self.parse()?;
                expect!(self.parser => TokenKind::RParen);
                Ok(expr)
            },
            TokenKind::Minus => {
                let item = Box::new(self.parse()?);
                Ok(FormulaExpr::Unary{
                    op: UnaryOperator::Neg,
                    item
                })
            },
            TokenKind::Lowpass => {
                expect!(self.parser => TokenKind::LParen);
                let value = Box::new(self.parse()?);
                expect!(self.parser => TokenKind::Comma);
                let cutoff = Box::new(self.parse()?);
                expect!(self.parser => TokenKind::Comma);
                let resonance = Box::new(self.parse()?);
                expect!(self.parser => TokenKind::RParen);
                Ok(FormulaExpr::Function(Function::Lowpass {
                    value, cutoff, resonance
                }))
            },
            TokenKind::Tanh => {
                expect!(self.parser => TokenKind::LParen);
                let value = Box::new(self.parse()?);
                expect!(self.parser => TokenKind::RParen);
                Ok(FormulaExpr::Function(Function::Tanh {
                    value
                }))
            },
            TokenKind::Sine => {
                expect!(self.parser => TokenKind::LParen);
                let value = Box::new(self.parse()?);
                expect!(self.parser => TokenKind::RParen);
                Ok(FormulaExpr::Function(Function::Sine {
                    value
                }))
            },
            t => Err(self.new_error(t))
        }
    }

    pub fn infix(&mut self, tok: TokenKind, left: FormulaExpr) -> Result<FormulaExpr, ParseError> {
        operator!(binary self, tok, left => 
            TokenKind::Minus => Operator::Minus,
            TokenKind::Asterisk => Operator::Times,
            TokenKind::Plus => Operator::Plus,
        )
    }

    fn infix_query(tok: &TokenKind) -> Option<Query> {
        // Lowest available precedence is 1 !!!
        match tok {
            TokenKind::Minus => Some(Query::new(5, false)),
            TokenKind::Plus => Some(Query::new(5, false)),
            TokenKind::Asterisk => Some(Query::new(6, false)),
            _ => None
        }
    }

    pub fn advance(&mut self) -> Result<TokenKind, ParseError> {
        self.parser.advance().ok_or(self.parser.new_error(ParseErrorKind::UnexpectedEnd))?
    }

    pub fn peek(&mut self) -> Option<&TokenKind> {
        self.parser.peek()
    }

    pub fn parse(&mut self) -> Result<FormulaExpr, ParseError> {
        self.parse_expression(0)
    }

    pub fn parse_expression(&mut self, precedence: u8) -> Result<FormulaExpr, ParseError> {
        let token = self.advance()?;

        let mut left = self.prefix(token)?;

        while let Some(token) = self.peek() {
            // println!("{:?} {} {:?}", token, precedence, FormulaParser::infix_query(token));
            if let Some(query) = FormulaParser::infix_query(token) {
                if precedence < query.precedence {
                    let token = self.advance()?;
                    left = self.infix(token, left)?
                } else {
                    break
                }
            } else {
                break
            }
        }

        Ok(left)
    }
}