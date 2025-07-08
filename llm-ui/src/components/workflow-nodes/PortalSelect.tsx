import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  MenuItem,
  ClickAwayListener,
  Popper,
  MenuList,
  Chip,
  FormControl,
  InputLabel,
  OutlinedInput,
  InputAdornment,
  IconButton,
} from '@mui/material';
import { ArrowDropDown as ArrowDropDownIcon } from '@mui/icons-material';

interface PortalSelectProps {
  value: string | string[];
  onChange: (value: string | string[]) => void;
  options: { value: string; label: string }[];
  label: string;
  multiple?: boolean;
  size?: 'small' | 'medium';
  fullWidth?: boolean;
  renderValue?: (selected: string | string[]) => React.ReactNode;
}

const PortalSelect: React.FC<PortalSelectProps> = ({
  value,
  onChange,
  options,
  label,
  multiple = false,
  size = 'small',
  fullWidth = true,
  renderValue,
}) => {
  const [open, setOpen] = useState(false);
  const anchorRef = useRef<HTMLDivElement>(null);

  const handleToggle = (event: React.MouseEvent) => {
    event.stopPropagation();
    setOpen((prevOpen) => !prevOpen);
  };

  const handleClose = (event: Event | React.SyntheticEvent) => {
    if (anchorRef.current && anchorRef.current.contains(event.target as HTMLElement)) {
      return;
    }
    setOpen(false);
  };

  const handleSelect = (selectedValue: string) => {
    if (multiple) {
      const currentValues = value as string[];
      if (currentValues.includes(selectedValue)) {
        onChange(currentValues.filter((v) => v !== selectedValue));
      } else {
        onChange([...currentValues, selectedValue]);
      }
    } else {
      onChange(selectedValue);
      setOpen(false);
    }
  };

  const getDisplayValue = () => {
    if (renderValue) {
      return renderValue(value);
    }

    if (multiple) {
      const selectedValues = value as string[];
      if (selectedValues.length === 0) return '';
      return (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
          {selectedValues.map((val) => {
            const option = options.find((o) => o.value === val);
            return <Chip key={val} label={option?.label || val} size="small" />;
          })}
        </Box>
      );
    } else {
      const option = options.find((o) => o.value === value);
      return option?.label || '';
    }
  };

  return (
    <>
      <FormControl size={size} fullWidth={fullWidth} ref={anchorRef}>
        <InputLabel shrink>{label}</InputLabel>
        <OutlinedInput
          value=""
          onClick={handleToggle}
          readOnly
          notched
          label={label}
          sx={{ cursor: 'pointer' }}
          endAdornment={
            <InputAdornment position="end">
              <IconButton edge="end" size="small">
                <ArrowDropDownIcon />
              </IconButton>
            </InputAdornment>
          }
          inputComponent={() => (
            <Box sx={{ display: 'flex', alignItems: 'center', px: 1.5, py: 0.5, minHeight: 32 }}>
              {getDisplayValue()}
            </Box>
          )}
        />
      </FormControl>

      <Popper
        open={open}
        anchorEl={anchorRef.current}
        placement="bottom-start"
        style={{ zIndex: 10000, width: anchorRef.current?.offsetWidth }}
        disablePortal={false}
      >
        <Paper elevation={8} sx={{ mt: 1, maxHeight: 300, overflow: 'auto' }}>
          <ClickAwayListener onClickAway={handleClose}>
            <MenuList>
              {options.map((option) => {
                const isSelected = multiple
                  ? (value as string[]).includes(option.value)
                  : value === option.value;
                
                return (
                  <MenuItem
                    key={option.value}
                    selected={isSelected}
                    onClick={() => handleSelect(option.value)}
                    sx={{
                      backgroundColor: isSelected ? 'action.selected' : undefined,
                    }}
                  >
                    {option.label}
                  </MenuItem>
                );
              })}
            </MenuList>
          </ClickAwayListener>
        </Paper>
      </Popper>
    </>
  );
};

export default PortalSelect;