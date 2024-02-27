use faer::{FaerMat, Mat};
use rustler::types::atom::nil;
use rustler::{Atom, Binary, Env, NifStruct, NifTuple, OwnedBinary};
use std::fmt;
use std::fmt::Debug;

mod nx_atoms {
    rustler::atoms! {
        f,
    }
}

#[derive(NifTuple)]
struct QR<'a> {
    q: Binary<'a>,
    r: Binary<'a>,
}

#[derive(NifTuple)]
struct QRTensor<'a> {
    q: NxTensor<'a>,
    r: NxTensor<'a>,
}

#[derive(Debug, NifTuple, Clone, Copy)]
struct NxType {
    kind: Atom,
    size: u8,
}

// Shape should be a n sized tuple
// but for now only support 2d tensors
#[derive(Debug, NifTuple)]
struct NxShape {
    row: usize,
    col: usize,
}

#[derive(NifStruct)]
#[module = "Nx.BinaryBackend"]
struct NxBinaryBackend<'a> {
    state: Binary<'a>,
}

#[derive(NifStruct)]
#[module = "Nx.Tensor"]
struct NxTensor<'a> {
    data: NxBinaryBackend<'a>,
    r#type: NxType,
    shape: NxShape,
    names: Vec<Atom>,
    vectorized_axes: Vec<Atom>,
}

impl Debug for NxTensor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NxTensor")
            .field("type", &self.r#type)
            .field("shape", &self.shape)
            .finish()
    }
}

fn binary_to_mat(bin: Binary, nrow: usize, ncol: usize) -> Mat<f64> {
    let mut mat: Mat<f64> = Mat::zeros(nrow, ncol);
    for (i, n) in bin.chunks(8).enumerate() {
        let c1: [u8; 8] = n[0..8].try_into().unwrap();
        let row = i / ncol;
        let col = i % ncol;
        mat.write(row, col, f64::from_le_bytes(c1))
    }
    mat
}

fn mat_to_binary(mat: &Mat<f64>) -> OwnedBinary {
    let nrow = mat.nrows();
    let ncol = mat.ncols();
    let mut owned_binary = OwnedBinary::new(8 * ncol * nrow).unwrap();
    for (i, byte) in owned_binary.chunks_exact_mut(8).enumerate() {
        let row = i / ncol;
        let col = i % ncol;
        byte.copy_from_slice(f64::to_le_bytes(mat.read(row, col)).as_slice());
    }
    owned_binary
}

#[rustler::nif]
fn qr_binary<'a>(env: Env<'a>, tensor: Binary, nrow: usize, ncol: usize) -> QR<'a> {
    //let mut result = OwnedBinary::new(tensor.len()).ok_or(Error::Term(Box::new("no mem")))?;
    //let mut mat: Mat<f64> = Mat::with_capacity(nrow, ncol);
    let mat = binary_to_mat(tensor, nrow, ncol);
    let qr = mat.qr();
    let q = qr.compute_thin_q();
    let r = qr.compute_thin_r();

    let result_q = mat_to_binary(&q);
    let result_r = mat_to_binary(&r);

    let result_qr = QR {
        q: Binary::from_owned(result_q, env),
        r: Binary::from_owned(result_r, env),
    };
    result_qr
}

#[rustler::nif]
fn qr_tensor<'a>(env: Env<'a>, tensor: NxTensor) -> QRTensor<'a> {
    let nrow = tensor.shape.row;
    let ncol = tensor.shape.col;
    let mat = binary_to_mat(tensor.data.state, nrow, ncol);
    let mat_qr = mat.qr();
    let q = mat_qr.compute_thin_q();
    let r = mat_qr.compute_thin_r();

    let result_q = mat_to_binary(&q);
    let result_r = mat_to_binary(&r);

    let q_tensor = NxTensor {
        r#type: tensor.r#type,
        shape: NxShape {
            row: nrow,
            col: ncol,
        },
        data: NxBinaryBackend {
            state: Binary::from_owned(result_q, env),
        },
        names: tensor.names.clone(),
        vectorized_axes: tensor.vectorized_axes.clone(),
    };

    let r_tensor = NxTensor {
        r#type: tensor.r#type,
        shape: NxShape {
            row: ncol,
            col: ncol,
        },
        data: NxBinaryBackend {
            state: Binary::from_owned(result_r, env),
        },
        names: tensor.names.clone(),
        vectorized_axes: tensor.vectorized_axes.clone(),
    };

    QRTensor {
        q: q_tensor,
        r: r_tensor,
    }
}

#[rustler::nif]
fn qr_binary_tensor<'a>(env: Env<'a>, tensor: Binary, nrow: usize, ncol: usize) -> QRTensor<'a> {
    let mat = binary_to_mat(tensor, nrow, ncol);
    let mat_qr = mat.qr();
    let q = mat_qr.compute_thin_q();
    let r = mat_qr.compute_thin_r();

    let result_q = mat_to_binary(&q);
    let result_r = mat_to_binary(&r);

    let t_type = NxType {
        kind: nx_atoms::f(),
        size: 64,
    };

    let q_tensor = NxTensor {
        r#type: t_type,
        shape: NxShape {
            row: nrow,
            col: ncol,
        },
        data: NxBinaryBackend {
            state: Binary::from_owned(result_q, env),
        },
        names: vec![nil(), nil()],
        vectorized_axes: vec![],
    };

    let r_tensor = NxTensor {
        r#type: t_type,
        shape: NxShape {
            row: ncol,
            col: ncol,
        },
        data: NxBinaryBackend {
            state: Binary::from_owned(result_r, env),
        },
        names: vec![nil(), nil()],
        vectorized_axes: vec![],
    };

    QRTensor {
        q: q_tensor,
        r: r_tensor,
    }
}

rustler::init!(
    "Elixir.NxRSBackend.RS",
    [qr_binary, qr_tensor, qr_binary_tensor]
);
