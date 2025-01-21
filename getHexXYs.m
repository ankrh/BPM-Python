function [x , y] = getHexXYs(N,pitch)
  x = NaN(N,1);
  y = NaN(N,1);
  x(1) = 0; % x of the center core
  y(1) = 0; % y of the center core
  shellSideIdx = 1; % Which side of the hex are we filling?
  shellSideCoreIdx = 0; % How many cores on this side have been filled so far?
  shellNum = 1; % Which shell are we in? The center core is not counted as a shell.
  for iC = 2:N
    if shellSideCoreIdx == 0 % If this is the first core in this shell
      x(iC) = shellNum*pitch;
      y(iC) = 0;
    else % Find new core position by adding onto the previous core's position
      x(iC) = x(iC-1) + pitch*cos(shellSideIdx*pi/3 + pi/3);
      y(iC) = y(iC-1) + pitch*sin(shellSideIdx*pi/3 + pi/3);
    end

    if shellSideCoreIdx == shellNum % If this side has been filled
      shellSideIdx = shellSideIdx + 1;
      shellSideCoreIdx = 1;
    else % Continue filling this side
      shellSideCoreIdx = shellSideCoreIdx + 1;
    end

    if shellSideCoreIdx == shellNum && shellSideIdx == 6 % Last core on last side would be a replicate of the first one drawn in this shell, so skip
      shellNum = shellNum + 1;
      shellSideIdx = 1;
      shellSideCoreIdx = 0;
    end
  end
end