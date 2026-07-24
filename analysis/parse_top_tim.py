from pathlib import Path
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--top-tim", type=Path, required=True, help="dir with top.tim")
opts = parser.parse_args()

state = "START"


def interesting_du_line(line):
    for s in ["DP16KD", "MULT18X18D", "ALU54B", "TRELLIS_FF", "TRELLIS_COMB"]:
        if s in line:
            # DU Info: 	              DP16KD:      21/     56    37%
            parts = line.split()
            assert len(parts) == 5, line
            return f"{parts[1]}{parts[4]}"
    return None


clocks = ["$glbnet$audio_clk", "$glbnet$clk"]
timings = {}
du_info_strs = []

fname = opts.top_tim
if not str(fname).endswith("top.tim"):
    fname = fname / "top.tim"

with open(fname, "r") as f:
    for line in f.readlines():
        line = line.strip()

        if line.startswith("Info: Max frequency for clock"):
            for clk in clocks:
                if clk in line:
                    timings[clk] = line
                    break

        elif line.startswith("Info: Router1 time"):
            time_str = line.split(" ")[-1]
            assert time_str.endswith("s")
            total_seconds = float(time_str[:-1])
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            print(f"Info: Router1 time: {hours}h {minutes:02d}m {seconds:02d}s")

        else:
            match state:
                case "START":
                    if line == "Info: Device utilisation:":
                        state = "DEVICE_UTIL"
                case "DEVICE_UTIL":
                    if len(line) == 0:
                        print("  ".join(du_info_strs))
                        state = "TIMING"
                    else:
                        line = interesting_du_line(line)
                        if line:
                            du_info_strs.append(line)

# print final clock timings
for clk in clocks:
    print(timings[clk].replace("Max frequency for clock", ""))
