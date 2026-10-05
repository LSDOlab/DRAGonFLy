// Mimic ParaView's VTX (vtkADIOS2VTXReader) UnstructuredGrid fill for one
// step: concatenate all writer blocks, offsetting each block's connectivity
// by the number of points in the preceding blocks ("squashed blocks"), as
// VTK's IO/ADIOS2/VTX/schema/vtk/VTXvtkVTU.cxx does. Points the blocks share
// are not merged, which is what makes MPI partition boundaries visible.
// Dumps raw arrays to <out>/ for analysis with VTK in Python (replay.py).
//
// usage: vtxdump <file.bp> <outdir> [step]
// build: c++ -std=c++17 vtxdump.cpp -I$PREFIX/include -L$PREFIX/lib \
//            -ladios2_cxx -ladios2_core -Wl,-rpath,$PREFIX/lib -o vtxdump
#include <adios2.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template <typename T>
std::vector<T> read_all_blocks(adios2::IO& io, adios2::Engine& eng, const std::string& name,
                               std::vector<size_t>& block_rows, size_t& ncols)
{
  auto var = io.InquireVariable<T>(name);
  if (!var)
    throw std::runtime_error("missing variable " + name);
  auto info = eng.BlocksInfo(var, eng.CurrentStep());
  std::vector<T> out;
  block_rows.clear();
  for (auto& b : info)
  {
    var.SetBlockSelection(b.BlockID);
    std::vector<T> buf;
    eng.Get(var, buf, adios2::Mode::Sync);
    size_t rows = b.Count.empty() ? 1 : b.Count[0];
    ncols = (b.Count.size() > 1) ? b.Count[1] : 1;
    block_rows.push_back(rows);
    out.insert(out.end(), buf.begin(), buf.end());
  }
  return out;
}

template <typename T>
void dump(const std::string& path, const std::vector<T>& v)
{
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}

int main(int argc, char** argv)
{
  std::string bp = argv[1], out = argv[2];
  size_t want_step = argc > 3 ? std::stoul(argv[3]) : 0;
  adios2::ADIOS adios;
  adios2::IO io = adios.DeclareIO("r");
  io.SetEngine("BP5");
  adios2::Engine eng = io.Open(bp, adios2::Mode::Read);
  std::ofstream meta(out + "/meta.txt");
  size_t step = 0;
  while (eng.BeginStep() == adios2::StepStatus::OK)
  {
    if (step != want_step)
    {
      eng.EndStep();
      ++step;
      continue;
    }
    std::vector<size_t> grows, crows, rows;
    size_t gcols = 0, ccols = 0, ncols = 0;
    auto geom = read_all_blocks<double>(io, eng, "geometry", grows, gcols);
    auto conn = read_all_blocks<int64_t>(io, eng, "connectivity", crows, ccols);
    // squash: offset connectivity of block b by sum of preceding point counts
    size_t lin = 0;
    int64_t off = 0;
    for (size_t b = 0; b < crows.size(); ++b)
    {
      for (size_t e = 0; e < crows[b]; ++e)
      {
        int64_t np = conn[lin];
        for (int64_t p = 0; p < np; ++p)
          conn[lin + 1 + p] += off;
        lin += np + 1;
      }
      off += grows[b];
    }
    dump(out + "/geometry.bin", geom);
    dump(out + "/connectivity.bin", conn);
    meta << "nblocks " << grows.size() << "\n";
    meta << "geometry " << geom.size() / gcols << " " << gcols << "\n";
    meta << "connectivity " << conn.size() / ccols << " " << ccols << "\n";
    auto types = io.InquireVariable<uint32_t>("types");
    uint32_t t;
    eng.Get(types, t, adios2::Mode::Sync);
    meta << "type " << t << "\n";
    auto stepvar = io.InquireVariable<double>("step");
    double stepval;
    eng.Get(stepvar, stepval, adios2::Mode::Sync);
    meta << "stepvalue " << stepval << "\n";
    for (auto& [name, params] : io.AvailableVariables())
    {
      if (name == "geometry" || name == "connectivity" || name == "types" || name == "step"
          || name == "NumberOfNodes" || name == "NumberOfCells" || name == "NumberOfEntities"
          || name == "vtkOriginalPointIds")
        continue;
      std::string type = params.at("Type");
      if (type == "double")
      {
        auto v = read_all_blocks<double>(io, eng, name, rows, ncols);
        dump(out + "/" + name + ".bin", v);
        meta << "f64 " << name << " " << v.size() / ncols << " " << ncols << "\n";
      }
      else if (type == "uint8_t")
      {
        auto v = read_all_blocks<uint8_t>(io, eng, name, rows, ncols);
        dump(out + "/" + name + ".bin", v);
        meta << "u8 " << name << " " << v.size() / ncols << " " << ncols << "\n";
      }
    }
    eng.EndStep();
    break;
  }
  eng.Close();
  // vtk.xml attribute tells which arrays are cell data
  auto attr = io.InquireAttribute<std::string>("vtk.xml");
  if (attr)
  {
    std::ofstream x(out + "/vtk.xml");
    x << attr.Data()[0];
  }
  return 0;
}
