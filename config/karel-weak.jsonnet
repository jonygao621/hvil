local programs = ["cs106a:0", "cs106a:3a", "cs106a:3b"];

{
  ["data/karel-weak/%s.json" % [program]]: {
    env: {
      name: "Karel",
      program: program,
    },
    agent: {
      name: "Karel-Weak",
      levels: 0,
      branch: 0,
    },
    data: {
      train: {
        num_traces: 1000,
        dirname: "data/karel-weak/%s/train" % [program],
      },
      test: {
        num_traces: 100,
        dirname: "data/karel-weak/%s/test" % [program],
      },
    },
  }
  for program in programs
}
+
{
  ["train/karel-weak/%s-L%d.json" % [program, level]]: {
    env: {
      name: "Karel",
      program: program,
    },
    agent: {
      name: "Karel-Weak",
      levels: level,
      branch: 5,
      num_grids: 0,
    },
    train: {
      name: "Karel-Weak_%s_L%s" % [program, level],
      data: {
        train: {
          dirname: "data/karel-weak/%s/train" % [program],
          num_annotated: 0,
          num_unannotated: 1000,
          batch_size: 10,
          unannotated_throttling_steps: 0,
        },
        test: {
          dirname: "data/karel-weak/%s/test" % [program],
          num_traces: 100,
        },
      },
      device: "cpu",
      num_steps: 100000,
      eval_freq: 100,
      save_freq: 100,
      keep_save_freq: 1000,
    }
  }
  for program in programs
  for level in [0, 1, 2]
}
