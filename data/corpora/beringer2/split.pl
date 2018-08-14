#!/usr/bin/env perl
#
use strict;
# process_file('broken_chords.abc', 'broken_chords');
process_file('arpeggios.abc', 'arpeggios');
#process_file('scales.abc', 'scales');

sub print_to_file {
   my ($content, $file_path) = @_;
   $file_path =~ s/arpeggios\_//g;
   $file_path =~ s/broken\_chords\_//g;
   $file_path =~ s/scales\_//g;
   $file_path .= '.abcd';
   open OUTPUT, ">${file_path}" or die "bad open of $file_path";
print OUTPUT "\% abcDidactyl v5
\% abcD fingering 1: x\@x
\% Authority:  Beringer and Dunhill (1900)
\% Transcriber: David Randolph
\% Transcription date: 2016-09-23 11:38:01
\% These are complete fingerings, with any gaps filled in.
\% abcD fingering 2: x\@x
\% Authority:  Beringer and Dunhill (1900)
\% Transcriber: David Randolph
\% Transcription date: 2016-09-23 12:38:01
\% These are alternate fingerings, if specified, with gaps filled in. 
\% abcDidactyl END
";
   print OUTPUT $content; 
   close OUTPUT or die "bad close of $file_path";
}

sub process_file {
   my ($input_file, $output_dir) = @_;
   system "rm -r $output_dir/*";

   open INPUT, "< $input_file" or die "bad open of $input_file";
   my $eg = '';
   my $file_name = '';
   foreach my $line (<INPUT>) {
      next if $line eq "\n";
      if ($line =~ /^T:\s*(.*)/) {
         $file_name = $1;
      }
      if ($line =~ /^X:\s*\d+/) {
         if ($eg) { 
            print_to_file($eg, "$output_dir/$file_name");
            $eg = '';
         }
      }
      $eg .= $line; 
   }
  
   print_to_file($eg, "$output_dir/$file_name");
   close INPUT or die "bad close of $input_file";
}


