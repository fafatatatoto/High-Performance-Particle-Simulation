# Parallel Particle Simulation: OpenMP and MPI

A 2-D short-range particle simulation, sped up two ways: with **OpenMP** on a single shared-memory node and with **MPI** across multiple nodes of a Slurm cluster.

| | Speedup | Hardware | Baseline |
|---|---|---|---|
| OpenMP | **17×** | 64 cores, 1 node | optimized serial code |
| MPI | **140×** | 96 processes, 4 nodes | optimized serial code |

Both versions were checked against reference outputs for correctness.

**Highlights**

- Cut the force computation from O(n²) to O(n) with a spatial grid of linked lists.
- Distributed the simulation with an MPI domain decomposition in which each process exchanges only boundary data with its two neighbors.
- Used strong- and weak-scaling experiments and a per-phase timing breakdown to find where time goes and why MPI scales further than OpenMP.

![Strong scaling, OpenMP and MPI](figures/strong_scaling_openmp_and_mpi.png)

---

## The problem

Particles move in a square box and push each other away, but only when two particles are closer than a fixed **cutoff** distance. Each of the 1,000 time steps computes the force on every particle and then moves all particles.

Checking every pair of particles costs O(n²) per step. Since only nearby particles interact, the work can be cut to O(n).

## Shared idea: spatial grid

The box is divided into square cells whose side is at least the cutoff. A particle can then only interact with particles in its own cell and the 8 cells around it, so each particle checks a small, nearly constant number of neighbors. This brings the cost per step from O(n²) down to O(n).

```
┌─────┬─────┬─────┬─────┬─────┐
│     │  ·  │     │ ·   │     │
├─────╆━━━━━┿━━━━━┿━━━━━╅─────┤
│  ·  ┃ ·   │   · │     ┃  ·  │      ● = the particle being updated
├─────╂─────┼─────┼─────╂─────┤      · = other particles
│     ┃   · │  ●  │  ·  ┃     │
├─────╂─────┼─────┼─────╂─────┤      Only the 3×3 cells in the bold box
│ ·   ┃     │ ·   │   · ┃     │      can be within the cutoff,
├─────╄━━━━━┿━━━━━┿━━━━━╃─────┤      so only those are checked.
│     │   · │     │     │  ·  │
└─────┴─────┴─────┴─────┴─────┘
```

Each cell keeps its particles in a **linked list**: one array stores the first particle of each cell, and another stores, for each particle, the next particle in the same cell.

```
head[cell]          next[particle]

cell 6  ──► P2 ──► P0 ──► (end)
cell 13 ──► P1 ──► (end)
cell 1  ──► P3 ──► (end)
other   ──► (end)              Most cells hold 0 or 1 particle.
```

## OpenMP version (shared memory)

- All threads share one particle array and one grid.
- The force loop and the move loop are split across threads with static scheduling. Computing forces only reads the grid, so it needs no synchronization.
- After moving, particles that changed cells are relinked into their new cell. Two threads can touch the same cell at once, so each cell has its own lock (**fine-grained locking**). A thread moving a particle locks both the old and the new cell, always in increasing index order, which rules out deadlock.

```
 thread 0   thread 1   thread 2   thread 3
     │          │          │          │
     ▼          ▼          ▼          ▼
┌──────────────────────────────────────────┐
│     one shared particle array + grid     │   every thread reads
│   (each cell has its own lock for moves) │   and writes the same memory
└──────────────────────────────────────────┘
```

## MPI version (distributed memory)

Each process has its own private memory, so particles have to be divided among processes, and anything a process needs from another must be sent as a message.

**Domain decomposition.** The box is split into horizontal strips of grid rows, one strip per process. Each particle is owned by exactly one process: the one whose strip contains it.

```
┌────────────────────────────┐
│ process 2                  │
│░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ ◄─ bottom row of process 2, copied to process 1 as ghosts
├────────────────────────────┤
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ ◄─ top row of process 1, copied to process 2 as ghosts
│ process 1                  │
│              ●  ─ ─ ─ ─ ─ ─│─► a particle that crosses the line
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│    is handed off to its new owner
├────────────────────────────┤
│░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│ process 0                  │
└────────────────────────────┘
```

**Each step:**

```mermaid
flowchart LR
    A[Ghost exchange<br/>with 2 neighbors] --> B[Rebuild grid<br/>own rows + ghost rows]
    B --> C[Compute forces<br/>owned particles only]
    C --> D[Move particles]
    D --> E[Handoff particles<br/>that left the strip]
    E --> A
```


1. **Ghost exchange.** Particles near a strip boundary feel forces from particles across it. Each process sends copies of the particles in its top and bottom rows to its two neighbors. Only positions are sent (16 bytes instead of 56), because computing a force only needs the neighbor's position.
2. **Grid construction.** Each process rebuilds the linked-list grid for its strip plus one ghost row on each side.
3. **Force computation and move,** only for owned particles. Ghost copies are read but never modified, so no forces need to be sent back.
4. **Handoff.** Particles that moved out of a strip are removed and sent, complete, to the neighbor that now owns them.

**Communication.** All exchanges are only with the two neighboring processes, using non-blocking `MPI_Isend` / `MPI_Irecv` followed by a single `MPI_Waitall`, which cannot deadlock. Because a receiver can't know in advance how many particles will arrive, each exchange first sends the count, then the data. To save output, all particles are collected on one process with `MPI_Gatherv` and restored to their original order in O(n).

## Hardware and setup

| | OpenMP runs | MPI runs |
|---|---|---|
| CPU per node | 2 × Intel Xeon Gold 6538Y+, 32 cores each (64 cores), up to 2.2 GHz | 2 × Intel Xeon Gold 6226, 12 cores each (24 cores), 2.7 GHz |
| Cache | 2 MB L2 per core, 60 MB L3 per socket | 1 MB L2 per core, 19.25 MB L3 per socket |
| Memory layout | 2 NUMA domains (one per socket) | 2 NUMA domains (one per socket) |
| Nodes used | 1 node, up to 64 threads | up to 4 nodes, 24 processes per node (96 total) |
| Interconnect | (single node) | InfiniBand |

No hyperthreading on either node type. Software: C++, OpenMP, MPI, CMake release build, Slurm. Every MPI timing is the average of 5 runs.

The two versions ran on **different node types**, so comparisons between OpenMP and MPI times are approximate.

## Results

### Runtime versus problem size

![Serial, OpenMP and MPI runtime, 10K to 320K particles](figures/serial_openmp_mpi_not_many_particles.png)

![OpenMP and MPI runtime, 100K to 3.2M particles](figures/openmp_mpi_many_particles.png)

On a log-log scale all versions have a slope close to 1: doubling the number of particles roughly doubles the runtime, confirming O(n) behavior. The parallel versions shift the line down, and MPI stays well below OpenMP as the problem grows to 3.2M particles.

### Strong scaling: same problem, more cores

The strong-scaling plot is shown at the top of this page.

| MPI processes | 1 | 2 | 4 | 8 | 16 | 24 | 48 | 96 |
|---|---|---|---|---|---|---|---|---|
| Time (s) | 74.28 | 32.79 | 16.84 | 8.47 | 3.42 | 2.04 | 1.04 | 0.53 |
| Speedup | 1.0 | 2.3 | 4.4 | 8.8 | 21.7 | 36.4 | 71.3 | 140.0 |

| OpenMP threads | serial | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Time (s) | 58.51 | 22.51 | 16.22 | 17.09 | 8.71 | 4.46 | 3.36 |
| Speedup | 1.0 | 2.6 | 3.6 | 3.4 | 6.7 | 13.1 | 17.4 |

MPI speedup is above linear. With one process, the data for 500K particles does not fit in cache, so each particle is slow to process. Split across many processes, each process's share fits in cache. Single-process runs show this directly: the time per particle per step grows from about 77 ns at 5K particles to about 149 ns at 500K, even though the algorithm is O(n).

OpenMP scales well below linear and stalls between 4 and 8 threads. See [Why OpenMP is slower than MPI](#why-openmp-is-slower-than-mpi) below.

### Weak scaling: problem grows with the cores

![Weak scaling, OpenMP and MPI](figures/weak_scaling_openmp_and_mpi.png)

| MPI processes | 1 | 2 | 4 | 8 | 16 | 24 | 48 | 96 |
|---|---|---|---|---|---|---|---|---|
| Time (s) | 0.83 | 0.80 | 0.86 | 0.83 | 0.88 | 0.85 | 1.00 | 1.02 |

With MPI, the problem grows 96× and runtime rises only 23% (81% efficiency). Runtime stays flat within one node and rises once processes span multiple nodes and messages travel over the network.

OpenMP's runtime, by contrast, grows several times over as threads and problem size increase together, reaching a scaled speedup of only 6.7× on 64 threads.

### Why OpenMP is slower than MPI

It is not only that MPI used more hardware. On a **single node**, MPI with 24 processes finishes 500K particles in 2.04 s, while OpenMP with 64 threads takes 3.36 s. The nodes differ (MPI ran on 2.7 GHz Xeon Gold 6226 nodes, OpenMP on 2.2 GHz Xeon Gold 6538Y+ nodes), so part of that gap may be the hardware, but MPI is faster with far fewer cores, and OpenMP also stops scaling well within its own node. The difference comes mostly from how each version uses memory:

| | OpenMP | MPI |
|---|---|---|
| Which particles a core works on | A slice of the particle array by **index**. Particles start in random order, so each thread's particles are scattered across the whole box | A strip of the **box**, so each process's particles sit next to each other in space |
| Memory each core touches | Neighbors can be anywhere in one large shared array, so most reads miss the cache | A small private array that fits in cache. This is also why MPI speedup is above linear |
| Synchronization | Every particle that changes cells takes two locks, and each step has three parallel loops that each end with a barrier where all threads wait | No locks. Processes only wait on their two neighbors, a few times per step |
| Memory bandwidth | All threads share one node's memory bandwidth | Spread over 4 nodes, the job gets 4 nodes' worth of bandwidth and cache |

The main cause is **locality**. MPI's domain decomposition forces each core to work on a compact region of space with its own data, while OpenMP threads all reach into the same large array. Locks and barriers add further overhead as threads are added, which matches OpenMP stalling between 4 and 8 threads. These explanations fit the measurements, but I did not profile cache misses or lock wait time directly.

An OpenMP version could recover much of this by giving each thread a region of the box instead of a range of particle indices, the same idea as MPI's decomposition, or by sorting particles by cell so that neighbors are close in memory.

### Where the time goes

Timing each phase on every process shows that computation dominates: about 90% of the runtime at 400K particles and above on 96 processes. Communication and waiting take a growing share as the number of processes rises. On small problems, communication becomes the limit. With 5K particles, going from 24 to 96 processes made the run only 1.3× faster, because each step is already close to the fixed latency of exchanging messages with neighbors.

## Design choices

**Linked list vs. counting sort for the grid.** I first stored each cell's particles contiguously using a counting sort, then switched to linked lists. The linked list was about 31% faster on one process (74 s vs. 107 s). Particles are sparse, about 0.2 per cell, so there are roughly 5× more cells than particles. Counting sort makes three passes over the cell array each step; the linked list makes one. Lists are so short that the linked list's usual drawback, scattered memory access, barely matters.

**1-D strips instead of 2-D blocks.** Strips mean only two neighbors and simple exchanges. The cost is that each strip spans the full width of the box, so the data each process exchanges grows as the problem grows. A 2-D block decomposition would keep that constant.

## What I would do next

- **2-D block decomposition**, to stop communication from growing with the problem size in weak scaling.
- **Overlap communication with computation.** Particles away from strip boundaries don't need ghosts, so their forces can be computed while ghost messages are in flight.
- **Fewer message rounds per step,** for example by merging the count and data messages, to reduce latency on small problems.
