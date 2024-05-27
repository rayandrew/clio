#!/usr/bin/gawk -f

# CC-BY-SA 4.0
# https://unix.stackexchange.com/a/398254

BEGIN{
  FPAT = "[0-9]+[dhms]";
  duration["s"] = 1;
  duration["m"] = 60;
  duration["h"] = duration["m"] * 60;
  duration["d"] = duration["h"] * 24;
}

{
  t=0;
  for (i=1; i<=NF; i++)
    t += $i * duration[substr($i, length($i))];
  print(t, $0);
}