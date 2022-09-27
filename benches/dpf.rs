#![feature(portable_simd)]

use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion, Throughput,
};
use dpf::pir::dpf_based::{self, ResponseScratchpad};
use dpf::pir::information_theoretic::{self, Query};
use dpf::{DpfKey, DPF_KEY_SIZE};
use std::simd::u64x8;

const LOG_BITS_IN_BYTE: usize = 3;

fn dpf_gen<const DEPTH: usize>() -> (DpfKey<DEPTH>, DpfKey<DEPTH>) {
    let hiding_point = 0;
    let mut point_val = [0u8; DPF_KEY_SIZE];
    point_val[0] = 1;
    let dpf_root_0 = [1u8; DPF_KEY_SIZE];
    let dpf_root_1 = [2u8; DPF_KEY_SIZE];
    DpfKey::gen(hiding_point, &point_val, dpf_root_0, dpf_root_1)
}

fn dpf_keygen_bench<const DEPTH: usize>(c: &mut Criterion) {
    let i = DEPTH;
    c.bench_with_input(BenchmarkId::new("dpf_keygen", DEPTH), &i, |b, &_s| {
        let hiding_point = 0;
        let mut point_val = [0u8; DPF_KEY_SIZE];
        point_val[0] = 1;
        let dpf_root_0 = [1u8; DPF_KEY_SIZE];
        let dpf_root_1 = [2u8; DPF_KEY_SIZE];
        b.iter(|| {
            black_box(DpfKey::<DEPTH>::gen(
                hiding_point,
                &point_val,
                dpf_root_0,
                dpf_root_1,
            ))
        });
    });
}

fn dpf_evalall_bench<const DEPTH: usize>(c: &mut Criterion) {
    let i = DEPTH;
    c.bench_with_input(BenchmarkId::new("dpf_evalall", DEPTH), &i, |b, &_s| {
        let (k_0, _) = dpf_gen::<DEPTH>();
        let mut output = vec![[0u8; DPF_KEY_SIZE]; 1 << DEPTH];
        let mut aux = vec![false; 1 << DEPTH];
        b.iter(|| black_box(k_0.eval_all_into(&mut output, &mut aux)));
    });
}

pub fn bench_dpf_gen(g: &mut Criterion) {
    dpf_keygen_bench::<3>(g);
    dpf_keygen_bench::<4>(g);
    dpf_keygen_bench::<5>(g);
    dpf_keygen_bench::<6>(g);
    dpf_keygen_bench::<7>(g);
    dpf_keygen_bench::<8>(g);
    dpf_keygen_bench::<9>(g);
    dpf_keygen_bench::<10>(g);
    dpf_keygen_bench::<11>(g);
    dpf_keygen_bench::<12>(g);
    dpf_keygen_bench::<13>(g);
    dpf_keygen_bench::<14>(g);
    dpf_keygen_bench::<15>(g);
    dpf_keygen_bench::<16>(g);
    dpf_keygen_bench::<17>(g);
    dpf_keygen_bench::<18>(g);
    dpf_keygen_bench::<19>(g);
    dpf_keygen_bench::<20>(g);
}
pub fn bench_dpf_evalall(g: &mut Criterion) {
    dpf_evalall_bench::<3>(g);
    dpf_evalall_bench::<4>(g);
    dpf_evalall_bench::<5>(g);
    dpf_evalall_bench::<6>(g);
    dpf_evalall_bench::<7>(g);
    dpf_evalall_bench::<8>(g);
    dpf_evalall_bench::<9>(g);
    dpf_evalall_bench::<10>(g);
    dpf_evalall_bench::<11>(g);
    dpf_evalall_bench::<12>(g);
    dpf_evalall_bench::<13>(g);
    dpf_evalall_bench::<14>(g);
    dpf_evalall_bench::<15>(g);
    dpf_evalall_bench::<16>(g);
    dpf_evalall_bench::<17>(g);
    dpf_evalall_bench::<18>(g);
    dpf_evalall_bench::<19>(g);
    dpf_evalall_bench::<20>(g);
}
pub fn bench_pir_single<const DPF_DEPTH: usize>(
    c: &mut BenchmarkGroup<WallTime>,
    db: &[u64x8],
    query_index: usize,
) {
    let dpf_root_0 = [1u8; DPF_KEY_SIZE];
    let dpf_root_1 = [2u8; DPF_KEY_SIZE];
    let (k_0, k_1) =
        dpf::pir::dpf_based::gen_query::<DPF_DEPTH>(query_index, dpf_root_0, dpf_root_1, db.len());

    let columns_amount = DPF_KEY_SIZE << (LOG_BITS_IN_BYTE + DPF_DEPTH);
    let mut output_0 = vec![u64x8::default(); db.len() / columns_amount];
    let mut output_1 = vec![u64x8::default(); db.len() / columns_amount];
    let eval_0 = k_0.eval_all();
    let eval_1 = k_1.eval_all();
    let mut scratch_0 = ResponseScratchpad::default();
    let mut scratch_1 = ResponseScratchpad::default();
    let query_0 = Query::new(unsafe {
        std::slice::from_raw_parts(eval_0[..].as_ptr().cast(), eval_0.len() * DPF_KEY_SIZE)
    });
    let query_1 = Query::new(unsafe {
        std::slice::from_raw_parts(eval_1[..].as_ptr().cast(), eval_1.len() * DPF_KEY_SIZE)
    });
    c.throughput(Throughput::Bytes(
        u64::try_from(db.len() * std::mem::size_of::<u64x8>()).unwrap(),
    ));
    let log_column_size_in_bytes = 26 - DPF_DEPTH;
    c.bench_with_input(
        BenchmarkId::new("information_theoretic_pir", log_column_size_in_bytes),
        &log_column_size_in_bytes,
        |b, _| {
            b.iter(|| {
                information_theoretic::answer_query_into(db, query_0, &mut output_0);
                information_theoretic::answer_query_into(db, query_1, &mut output_1);
            });
        },
    );
    c.bench_with_input(
        BenchmarkId::new("dpf_pir_bfs", log_column_size_in_bytes),
        &log_column_size_in_bytes,
        |b, _| {
            b.iter(|| {
                dpf_based::answer_query_into_with_scratchpad(
                    db,
                    &k_0,
                    &mut output_0,
                    &mut scratch_0,
                );
                dpf_based::answer_query_into_with_scratchpad(
                    db,
                    &k_1,
                    &mut output_1,
                    &mut scratch_1,
                );
            });
        },
    );
    c.bench_with_input(
        BenchmarkId::new("dpf_pir_dfs", log_column_size_in_bytes),
        &log_column_size_in_bytes,
        |b, _| {
            b.iter(|| {
                dpf_based::answer_query_into(db, &k_0, &mut output_0);
                dpf_based::answer_query_into(db, &k_1, &mut output_1);
            });
        },
    );
}
pub fn bench_pir(c: &mut Criterion) {
    const LOG_DB_SZ: usize = 33;
    const DB_SZ: usize = 1 << LOG_DB_SZ;
    const QUERY_INDEX: usize = 257;
    const BITS_IN_BYTE: usize = 8;
    let mut g = c.benchmark_group("pir");
    let db: Vec<_> = (0..(DB_SZ / (std::mem::size_of::<u64x8>() * BITS_IN_BYTE)))
        .map(|i| u64x8::splat(i as u64))
        .collect();
    bench_pir_single::<1>(&mut g, &db, QUERY_INDEX);
    bench_pir_single::<2>(&mut g, &db, QUERY_INDEX);
    bench_pir_single::<3>(&mut g, &db, QUERY_INDEX);
    bench_pir_single::<4>(&mut g, &db, QUERY_INDEX);
    bench_pir_single::<5>(&mut g, &db, QUERY_INDEX);
    bench_pir_single::<6>(&mut g, &db, QUERY_INDEX);
    bench_pir_single::<7>(&mut g, &db, QUERY_INDEX);
    bench_pir_single::<8>(&mut g, &db, QUERY_INDEX);
    bench_pir_single::<9>(&mut g, &db, QUERY_INDEX);
    bench_pir_single::<10>(&mut g, &db, QUERY_INDEX);
    bench_pir_single::<11>(&mut g, &db, QUERY_INDEX);
    bench_pir_single::<12>(&mut g, &db, QUERY_INDEX);
    bench_pir_single::<13>(&mut g, &db, QUERY_INDEX);
    bench_pir_single::<14>(&mut g, &db, QUERY_INDEX);
    bench_pir_single::<15>(&mut g, &db, QUERY_INDEX);
    bench_pir_single::<16>(&mut g, &db, QUERY_INDEX);
    bench_pir_single::<17>(&mut g, &db, QUERY_INDEX);
    g.finish();
}

criterion_group!(benches, bench_pir, bench_dpf_gen, bench_dpf_evalall);
criterion_main!(benches);
